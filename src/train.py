import argparse
import cv2
import os
import torch
import numpy as np
import pandas as pd
from apex import amp
from datetime import datetime
from torch import nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import set_seed, write_log, mask2rle
from data import get_train_metadata, get_test_metadata, get_train_test_split, CloudDataset
from model import get_model, RAdam, DiceLoss, tta, post_process, load_model
from metric import EarlyStopping, AverageMeter, get_auc_scores, SegMeter
from loss import lovasz_hinge


parser = argparse.ArgumentParser(description='parameters for program')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--grad_accum', type=int, default=1)
parser.add_argument('--encoder', type=str, default='efficientnet-b3')
parser.add_argument('--decoder', type=str, default='unet')
parser.add_argument('--height', type=int, default=384)
parser.add_argument('--width', type=int, default=576)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--blackout', type=float, default=0.)
parser.add_argument('--lovasz_hinge', type=int, default=-1)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.deterministic = True
cudnn.benchmark = True

CLASS_TARGETS = ['Fish', 'Flower', 'Gravel', 'Sugar']

FOLD = args.fold
H = args.height
W = args.width
ENCODER_NAME = args.encoder
DECODER_NAME = args.decoder
BATCH_SIZE = args.batch_size
GRAD_ACCUM = args.grad_accum
BLACKOUT = args.blackout
LOVASZ_HINGE = args.lovasz_hinge

MODEL_NAME = '{}_{}_{}_{}'.format(DECODER_NAME, ENCODER_NAME, H, W)
if BLACKOUT:
    MODEL_NAME += '_blackout_p_{}'.format(BLACKOUT)
if LOVASZ_HINGE:
    MODEL_NAME += '_lovasz_hinge_{}'.format(LOVASZ_HINGE)
LOG_FILE = '../log/{}_fold_{}.txt'.format(MODEL_NAME, FOLD)
MODEL_CHECKPOINT = '../model/{}_fold_{}.pth'.format(MODEL_NAME, FOLD)
SUB_FILE = '../submission/{}_fold_{}.csv'.format(MODEL_NAME, FOLD)

NUM_WORKERS = 8
MAX_EPOCH = 90
EARLY_STOP = 7
INITIAL_LR = 1e-3
W1, W2, W3 = 1, 1, 1
UNET_ATTEN = False
FPN_DROPOUT = .2

TRAIN = 1
PREDICT = 0

set_seed(42)
train_running_message = '\r-- epoch {:>3} - iter {:>5} - secs per batch {:4.2f} - train clf loss {:5.4f} seg loss {:5.4f} dice loss {:5.4f}'
train_epoch_end_message = '\r-- epoch {:>3} - secs total {:4.2f} - train auc {:5.4f}  clf loss {:5.4f} - best dice score is {:5.4f} with threshold value {} and details:'
valid_running_message = '\r-- epoch {:>3} - iter {:>5} - secs per batch {:4.2f} - valid clf loss {:5.4f} seg loss {:5.4f} dice loss {:5.4f}'
valid_epoch_end_message = '\r-- epoch {:>3} - secs total {:4.2f} - valid auc {:5.4f}  clf loss {:5.4f} - best dice score is {:5.4f} with threshold value {} and details:'
auc_message = '-- per label auc score: Fish {:5.4f} Flower {:5.4f} Gravel {:5.4f} Sugar {:5.4f}'


metadata = get_train_metadata()
folds = get_train_test_split(metadata)
train_idx, valid_idx = folds[FOLD]
train_meta = metadata.iloc[train_idx].copy()
valid_meta = metadata.iloc[valid_idx].copy()
train_ds = CloudDataset(train_meta, mode='train', blackout_p=BLACKOUT, H=H, W=W)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_ds = CloudDataset(valid_meta, mode='valid')
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = get_model(ENCODER_NAME, DECODER_NAME, UNET_ATTEN, FPN_DROPOUT).cuda()
clf_criterion = nn.BCEWithLogitsLoss()
seg_criterion_1 = nn.BCEWithLogitsLoss()
seg_criterion_2 = DiceLoss()
optimizer = RAdam(model.parameters(), lr=INITIAL_LR)  # Adam(model.parameters(), lr=INITIAL_LR)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, verbose=1, min_lr=1e-7, factor=.1)
earlystop = EarlyStopping(mode='max', patience=EARLY_STOP, percentage=False)
model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)


if TRAIN:
    best_score, best_epoch, history = 0, 0, pd.DataFrame()
    for epoch in range(MAX_EPOCH):
        if LOVASZ_HINGE and epoch + 1 >= LOVASZ_HINGE:
            seg_criterion_1 = lovasz_hinge
        tt0 = datetime.now()
        clf_loss_meter = AverageMeter()
        seg_loss_1_meter = AverageMeter()
        seg_loss_2_meter = AverageMeter()
        seg_metric_meter = SegMeter()
        train_clf_labels, train_clf_preds = [], []

        optimizer.zero_grad()
        for it, (images, classes, masks) in enumerate(train_dl, 1):
            t0 = datetime.now()

            model = model.train()
            images, classes, masks = images.cuda(), classes.cuda().float(), masks.cuda()
            clf_logits, seg_logits = model(images)

            clf_loss = clf_criterion(clf_logits, classes)
            seg_loss_1 = seg_criterion_1(seg_logits, masks)
            seg_loss_2 = seg_criterion_2(seg_logits, masks)
            loss = W1 * clf_loss + W2 * seg_loss_1 + W3 * seg_loss_2

            if GRAD_ACCUM > 1:
                loss = loss / GRAD_ACCUM
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if it % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_clf_labels.append(classes.cpu().detach().numpy())
            train_clf_preds.append(clf_logits.sigmoid().cpu().detach().numpy())

            clf_loss_meter.update(clf_loss.cpu().detach().numpy())
            seg_loss_1_meter.update(seg_loss_1.cpu().detach().numpy())
            seg_loss_2_meter.update(seg_loss_2.cpu().detach().numpy())
            seg_metric_meter.update(masks, clf_logits, seg_logits)

            dt = (datetime.now() - t0).total_seconds()
            message = train_running_message.format(
                epoch + 1, it, dt,
                clf_loss_meter.avg, seg_loss_1_meter.avg, seg_loss_2_meter.avg
            )
            print(message, end='', flush=True)

        train_clf_preds = np.vstack(train_clf_preds)
        train_clf_labels = np.vstack(train_clf_labels)
        scores = get_auc_scores(train_clf_labels, train_clf_preds)
        clf_thresholds, seg_thresholds, dice_score, results = seg_metric_meter.get_scores()
        print('\r\n', end='')
        write_log(message, LOG_FILE)
        dtt = (datetime.now() - tt0).total_seconds()
        message = train_epoch_end_message.format(
            epoch + 1, dtt, np.mean(scores), clf_loss_meter.avg, dice_score,
            ' '.join([str(val) for val in clf_thresholds + seg_thresholds])
        )
        print(message)
        write_log(message, LOG_FILE)
        message = auc_message.format(*scores)
        print(message)
        write_log(message, LOG_FILE)
        print(results)
        write_log(str(results) + '\n', LOG_FILE)
        results['phase'] = 'train'
        results['epoch'] = epoch + 1
        history = pd.concat([history, results], 0)

        tt0 = datetime.now()
        clf_loss_meter = AverageMeter()
        seg_loss_1_meter = AverageMeter()
        seg_loss_2_meter = AverageMeter()
        seg_focal_loss_meter = AverageMeter()
        seg_metric_meter = SegMeter()
        valid_clf_preds = []

        for it, (images, classes, masks) in enumerate(valid_dl, 1):
            t0 = datetime.now()

            model = model.eval()
            with torch.no_grad():
                images, classes, masks = images.cuda(), classes.cuda().float(), masks.cuda()
                clf_logits, seg_logits = tta(images, model)
                clf_loss = clf_criterion(clf_logits, classes)
                seg_loss_1 = seg_criterion_1(seg_logits, masks)
                seg_loss_2 = seg_criterion_2(seg_logits, masks)

                valid_clf_preds.append(clf_logits.sigmoid().cpu().detach().numpy())

                clf_loss_meter.update(clf_loss.cpu().detach().numpy())
                seg_loss_1_meter.update(seg_loss_1.cpu().detach().numpy())
                seg_loss_2_meter.update(seg_loss_2.cpu().detach().numpy())
                seg_metric_meter.update(masks, clf_logits, seg_logits)

                dt = (datetime.now() - t0).total_seconds()
                message = valid_running_message.format(
                    epoch + 1, it, dt,
                    clf_loss_meter.avg, seg_loss_1_meter.avg, seg_loss_2_meter.avg,
                )
                print(message, end='', flush=True)
        valid_clf_preds = np.vstack(valid_clf_preds)
        scores = get_auc_scores(valid_ds.labels, valid_clf_preds)
        clf_thresholds, seg_thresholds, dice_score, results = seg_metric_meter.get_scores()
        print('\r\n', end='')
        write_log(message, LOG_FILE)
        dtt = (datetime.now() - tt0).total_seconds()
        message = valid_epoch_end_message.format(
            epoch + 1, dtt, np.mean(scores), clf_loss_meter.avg, dice_score,
            ' '.join([str(val) for val in clf_thresholds + seg_thresholds])
        )
        print(message)
        write_log(message, LOG_FILE)
        message = auc_message.format(*scores)
        print(message)
        write_log(message, LOG_FILE)
        print(results)
        write_log(str(results) + '\n', LOG_FILE)
        results['phase'] = 'valid'
        results['epoch'] = epoch + 1
        history = pd.concat([history, results], 0)

        if dice_score > best_score:
            best_score = dice_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_CHECKPOINT)

        scheduler.step(dice_score)
        earlystop.step(dice_score)

        if earlystop.stop:
            break

        message = 'best_epoch so far was at {} and dice score {}'.format(best_epoch, best_score)
        print(message)
        write_log(message, LOG_FILE)

    message = 'done! best_epoch happened at {} and dice score {}'.format(best_epoch, best_score)
    print(message)
    write_log(message, LOG_FILE)
    history.to_csv('{}.csv'.format(LOG_FILE.replace('.txt', ''), index=False))

if PREDICT:
    model = load_model(MODEL_CHECKPOINT, ENCODER_NAME, DECODER_NAME, UNET_ATTEN, FPN_DROPOUT)
    test_metadata = get_test_metadata()
    test_ds = CloudDataset(test_metadata, mode='test')
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    CLF_THRESHOLD = .7
    SEG_THRESHOLD = .3
    predictions = []
    for idx, (fnames, images) in tqdm(enumerate(test_dl, 1), total=len(test_dl)):
        with torch.no_grad():
            logits, masks = tta(images.cuda(), model)
            masks = torch.sigmoid(masks).detach().cpu().numpy()
            probas = torch.sigmoid(logits).detach().cpu().numpy()
        for fname, mask, proba in zip(fnames, masks, probas):
            for i, label in enumerate(CLASS_TARGETS):
                proba_i = proba[i]
                mask_i = cv2.resize(mask[i, :], (525, 350), interpolation=cv2.INTER_LINEAR)
                pred = post_process(mask_i, proba_i, SEG_THRESHOLD, CLF_THRESHOLD)
                rle = mask2rle(pred)
                name = "{}_{}".format(fname, label)
                predictions.append([name, rle])

    sub = pd.DataFrame(predictions, columns=['Image_Label', 'EncodedPixels'])
    sub.to_csv(SUB_FILE.split('.csv')[0] + '_' + str(int(CLF_THRESHOLD * 10)) + '_' + str(int(SEG_THRESHOLD * 10)) + '.csv', index=False)

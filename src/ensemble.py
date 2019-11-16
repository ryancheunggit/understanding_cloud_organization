import cv2
import torch
import pandas as pd
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import mask2rle
from data import get_test_metadata, CloudDataset
from model import load_model, tta, post_process


cudnn.deterministic = True
cudnn.benchmark = True

MODEL_NAME = 'unet_efficientnet-b3_384_576'
SUB_FILE = '../submission/{}_ensemble.csv'.format(MODEL_NAME)

CLASS_TARGETS = ['Fish', 'Flower', 'Gravel', 'Sugar']

BATCH_SIZE = 32
NUM_WORKERS = 8
ENCODER_NAME = 'efficientnet-b3'  # 'se_resnext50_32x4d'
DECODER_NAME = 'unet'
UNET_ATTEN = False
FPN_DROPOUT = .2
models = [load_model(model_checkpoint, ENCODER_NAME, DECODER_NAME, UNET_ATTEN, FPN_DROPOUT)
          for model_checkpoint in ['../model/unet_efficientnet-b3_384_576_fold_{}.pth'.format(i) for i in range(5)]]
n = len(models)
test_metadata = get_test_metadata()
test_ds = CloudDataset(test_metadata, mode='test')
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

CLF_THRESHOLD = .7
SEG_THRESHOLD = .3
predictions = []
for idx, (fnames, images) in tqdm(enumerate(test_dl, 1), total=len(test_dl)):
    with torch.no_grad():
        images = images.cuda()
        for i, model in enumerate(models):
            if i == 0:
                logits, masks = tta(images, model)
                logits /= n
                masks /= n
            else:
                new_logits, new_masks = tta(images, model)
                logits += new_logits / n
                masks += new_masks / n

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

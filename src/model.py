import cv2
import math
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch import nn
from torch.optim.optimizer import Optimizer


UNET_CHANNELS = [256, 128, 64, 32, 16]
CNN_FEATURES = {
    'resnet34': 512,
    'resnet50': 2048,
    'efficientnet-b0': 320,
    'efficientnet-b1': 320,
    'efficientnet-b2': 352,
    'efficientnet-b3': 384,
    'efficientnet-b4': 448,
    'efficientnet-b5': 512,
    'se_resnext50_32x4d': 2048,
    'se_resnext101_32x4d': 4096,
}


class MultiTaskModel(nn.Module):
    def __init__(self, model, num_features, num_classes=4, topk=10):
        super(MultiTaskModel, self).__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.topk = topk

    def forward(self, x):
        features = self.encoder(x)
        masks = self.decoder(features)
        logits = torch.topk(masks.view(masks.shape[0], masks.shape[1], -1), self.topk)[0].mean(2)
        return logits, masks


def get_model(encoder_name='efficientnet-b3', decoder_name='unet', unet_atten=False, fpn_dropout=.2):
    model_constructor = smp.Unet if decoder_name == 'unet' else smp.FPN
    if decoder_name == 'unet':
        model = model_constructor(encoder_name=encoder_name, classes=4, activation=None,
                                  attention_type='scse' if unet_atten else None, decoder_channels=UNET_CHANNELS)
    elif decoder_name == 'fpn':
        model = model_constructor(encoder_name=encoder_name, classes=4, activation=None, dropout=fpn_dropout)
    cnn_features = CNN_FEATURES[encoder_name]
    model = MultiTaskModel(model, cnn_features, num_classes=4)
    return model


def load_model(checkpoint, encoder_name='efficientnet-b3', decoder_name='unet', unet_atten=False, fpn_dropout=.2, use_gpu=True):
    model = get_model(encoder_name=encoder_name, decoder_name=decoder_name, unet_atten=unet_atten, fpn_dropout=fpn_dropout)
    model_state = torch.load(checkpoint)
    model.load_state_dict(model_state)
    if use_gpu:
        model = model.cuda()
    model = model.eval()
    return model


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None]
                                                                                        for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, class_weights=None, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.class_weights = torch.from_numpy(np.array(class_weights)) if class_weights else None

    def forward(self, logits, labels):
        probas = logits.sigmoid()
        intersection = torch.sum(probas * labels, dim=[0, 2, 3])
        cardinality = torch.sum(probas + labels, dim=[0, 2, 3])
        dice_loss = 2. * intersection / (cardinality + self.epsilon)
        if self.class_weights:
            dice_loss = self.class_weights * dice_loss
        return (1 - dice_loss.mean())


def post_process(mask_i, proba_i, seg_threshold, clf_threshold, min_size=0):
    predictions = np.zeros((350, 525), np.float32)
    mask = (mask_i > seg_threshold).astype(np.float32)
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    if proba_i > clf_threshold:
        predictions = mask
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > min_size:
                predictions[p] = 1
    return predictions


def tta(images, model):
    with torch.no_grad():
        clf_logits, seg_logits = model(images)
        clf_logits_h, seg_logits_h = model(images.flip(2))
        clf_logits_v, seg_logits_v = model(images.flip(3))
        clf_logits_hv, seg_logits_hv = model(images.flip(2, 3))
        seg_logits_h = seg_logits_h.flip(2)
        seg_logits_v = seg_logits_v.flip(3)
        seg_logits_hv = seg_logits_hv.flip(2, 3)
        clf_logits = (clf_logits + clf_logits_h + clf_logits_v + clf_logits_hv) / 4.
        seg_logits = (seg_logits + seg_logits_h + seg_logits_v + seg_logits_hv) / 4.
    return clf_logits, seg_logits

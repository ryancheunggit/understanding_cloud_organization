import torch
import numpy as np
import pandas as pd
from operator import lt, gt
from sklearn.metrics import roc_auc_score


CLASS_TARGETS = ['Fish', 'Flower', 'Gravel', 'Sugar']
SEG_THRESHOLDS = [.3, .4, .5, .6, .7, .8, .9]
CLF_THRESHOLDS = [.3, .4, .5, .6, .7, .8, .9]


class EarlyStopping(object):
    """Monitoring an metric, flag when to stop training."""
    def __init__(self, mode='min', min_delta=0, percentage=False, patience=10, initial_bad=0, initial_best=np.nan):
        assert patience > 0, 'patience must be positive integer'
        assert mode in ['min', 'max'], 'mode must be either min or max'
        self.mode = mode
        self.patience = patience
        self.best = initial_best
        self.num_bad_epochs = initial_bad
        self.is_better = self._init_is_better(mode, min_delta, percentage)
        self._stop = False

    @property
    def stop(self):
        return self._stop

    def step(self, metric):
        if self.is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
        else:
            self.num_bad_epochs += 1

        if np.isnan(self.best) and (not np.isnan(metric)):
            self.num_bad_epochs = 0
            self.best = metric

        self._stop = self.num_bad_epochs >= self.patience

    def _init_is_better(self, mode, min_delta, percentage):
        comparator = lt if mode == 'min' else gt
        if not percentage:
            def _is_better(new, best):
                target = best - min_delta if mode == 'min' else best + min_delta
                return comparator(new, target)
        else:
            def _is_better(new, best):
                target = best * (1 - (min_delta / 100)) if mode == 'min' else best * (1 + (min_delta / 100))
                return comparator(new, target)
        return _is_better

    def __repr__(self):
        return '<EarlyStopping object with: mode - {} - num_bad_epochs - {} - patience - {} - best - {}>'.format(self.mode, self.num_bad_epochs, self.patience, self.best)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_auc_scores(labels, preds, num_labels=4):
    return [roc_auc_score(labels[:, i], preds[:, i]) for i in range(num_labels)]


def eval_dice_scores(logit, proba, mask, clf_threshold, seg_threshold):
    """
    proba: tensor of shape batch_size x # total_pixels
    mask: tensor of shape batch_size x # total_pixels
    threshold: float
    """
    with torch.no_grad():
        p = (proba * (logit > clf_threshold).float() > seg_threshold).float()
        t = (mask > .5).float()
        t_sum = t.sum(-1)  # batch_size x 1
        p_sum = p.sum(-1)  # batch_size x 1
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)
        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))
        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])
        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()
        num_neg_truth = len(neg_index)
        num_pos_truth = len(pos_index)
        num_neg_preds = len(torch.nonzero(p_sum == 0))
        num_pos_preds = len(torch.nonzero(p_sum >= 1))
    return {
            'dice': dice, 'dice_neg': dice_neg, 'dice_pos': dice_pos,
            'num_neg_truth': num_neg_truth, 'num_neg_preds': num_neg_preds,
            'num_pos_truth': num_pos_truth, 'num_pos_preds': num_pos_preds
        }


def get_seg_metrics(logits, probas, masks, seg_thresholds=SEG_THRESHOLDS, clf_thresholds=CLF_THRESHOLDS):
    """evaluate segmentation results at specified thresholds."""
    batch_size = len(masks)
    results = []
    for seg_threshold in seg_thresholds:
        for i, label in enumerate(CLASS_TARGETS):
            for clf_threshold in clf_thresholds:
                result = {'seg_threshold': seg_threshold, 'clf_threshold': clf_threshold, 'label': label}
                with torch.no_grad():
                    proba = probas[:, i, :, :].view(batch_size, -1)
                    mask = masks[:, i, :, :].view(batch_size, -1)
                    logit = logits[:, i].view(batch_size, -1)
                    assert proba.shape == mask.shape
                    result.update(eval_dice_scores(logit, proba, mask, clf_threshold, seg_threshold))
                results.append(result)
    return results


class SegMeter(object):
    def __init__(self):
        self.results = pd.DataFrame()

    def update(self, masks, clf_logits, seg_logits):
        with torch.no_grad():
            probas = torch.sigmoid(seg_logits)
        results = pd.DataFrame(get_seg_metrics(clf_logits, probas, masks))
        self.results = pd.concat([self.results, results], 0)

    # TODO: change according to competitio calculation
    def get_scores(self):
        summary = self.results.groupby(['clf_threshold', 'seg_threshold', 'label']).agg({
            'dice': 'mean', 'dice_neg': 'mean', 'dice_pos': 'mean',
            'num_pos_truth': 'sum', 'num_pos_preds': 'sum', 'num_neg_truth': 'sum', 'num_neg_preds': 'sum'
        }).reset_index()
        results = pd.DataFrame()
        best_clf_thresholds, best_seg_thresholds = [], []
        for label in CLASS_TARGETS:
            df = summary.loc[summary['label'] == label].copy()
            best = df.groupby(['clf_threshold', 'seg_threshold'])['dice'].mean().sort_values(ascending=False).reset_index(name='dice')
            best_clf_threshold = best['clf_threshold'][0]
            best_seg_threshold = best['seg_threshold'][0]
            best_clf_thresholds.append(best_clf_threshold)
            best_seg_thresholds.append(best_seg_threshold)
            threshold_results = df.loc[(df['clf_threshold'] == best_clf_threshold) & (df['seg_threshold'] == best_seg_threshold)].copy()
            results = pd.concat([results, threshold_results], 0)
        results.sort_values('label', inplace=True)
        dice_score = results['dice'].mean()
        return best_clf_thresholds, best_seg_thresholds, dice_score, results

import numpy as np

def calc_metrics(preds, preds_adv, y_gt, inds):
    acc_all = np.mean(preds[inds] == y_gt[inds])
    acc_all_adv = np.mean(preds_adv[inds] == y_gt[inds])
    return acc_all, acc_all

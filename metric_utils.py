import numpy as np


def calc_adv_acc(preds, preds_adv, y_test, test_inds):
    acc = np.mean(preds[test_inds] == y_test[test_inds])
    acc_adv = np.mean(preds_adv[test_inds] == y_test[test_inds])
    return acc, acc_adv

def calc_first_n_adv_acc(preds, preds_adv, y_test, test_inds, n):
    acc = np.mean(preds[test_inds][0:n] == y_test[test_inds][0:n])
    acc_adv = np.mean(preds_adv[test_inds][0:n] == y_test[test_inds][0:n])
    return acc, acc_adv

def calc_first_n_adv_acc_from_probs_summation(all_probs, all_probs_adv, y_test, test_inds, n):
    probs = all_probs.sum(axis=1)
    preds = probs.argmax(axis=1)
    probs_adv = all_probs_adv.sum(axis=1)
    preds_adv = probs_adv.argmax(axis=1)
    return calc_first_n_adv_acc(preds, preds_adv, y_test, test_inds, n)






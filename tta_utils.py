import os
import sys
import time
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit, jit
import torch
import torch.nn as nn
import scipy
from time import time
import inspect
from active_learning_project.global_vars import features_index, normal_features_list, adv_features_list, FEATURES_RANKS

rand_gen = np.random.RandomState(12345)
eps = 1e-10
PLOT = False

def to_features(func):
    """Decorator to fetch name of feature, normal features, and adv features"""
    def inner1(*args, **kwargs):
        global features_index, normal_features_list, adv_features_list
        begin = time()
        f_out = func(*args, **kwargs)
        features_index.append(f_out[0])
        normal_features_list.append(f_out[1])
        adv_features_list.append(f_out[2])
        end = time()
        print("Total time taken in {}: {}".format(func.__name__, end - begin))

    return inner1

def plot_ttas(x, x_adv, f2_inds, n_imgs=5, n_dist=10):
    """
    :param x: np.ndarray images
    :param x_adv: np.ndarray images
    :param f2_inds: list of images that were classified and attacked successfully.
    :param n_imgs: number of images to plot
    :param n_dist: number of distortions to plot
    :return: None. Plotting images
    """
    num_points = x.shape[1]
    assert num_points % n_dist == 0
    p_delta = int(num_points / n_dist)
    inds = rand_gen.choice([si for si in f2_inds if si < 100], n_imgs, replace=False)
    fig = plt.figure(figsize=(n_dist, 2 * n_imgs))
    for i in range(n_imgs):
        for p in range(n_dist):
            loc = n_dist * (2 * i) + p + 1
            fig.add_subplot(2 * n_imgs, n_dist, loc)
            plt.imshow(x[inds[i], p * p_delta])
            plt.axis('off')
            loc = n_dist * (2 * i + 1) + p + 1
            fig.add_subplot(2 * n_imgs, n_dist, loc)
            plt.imshow(x_adv[inds[i], p * p_delta])
            plt.axis('off')
    plt.tight_layout()
    plt.show()

def update_useful_stats(stats):
    """
    Computing useful statistics in one place to save computation time.
    :param stats: dictionary with some stats: 'preds' and 'losses'.
    :return: A dictionary (stats) containing useful statistics to be used to get later features.
    """
    test_size, num_points, num_classes = stats['preds'].shape

    # get softmax probabilities
    stats['probs'] = scipy.special.softmax(stats['preds'], axis=2)
    stats['probs_mean'] = stats['probs'].mean(axis=1)

    # get relative losses
    losses = stats['losses']
    rel_losses = np.zeros_like(losses)
    for k in range(test_size):
        rel_losses[k] = (losses[k] - losses[k, 0]) / losses[k, 0]
    stats['rel_losses'] = rel_losses

    # get hard predictions
    stats['y_ball_preds'] = stats['probs'].argmax(axis=2)

    # get all tta ranks (orders) that yielded different prediction than the original image
    stats['switch_ranks'] = []
    for k in range(test_size):
        rks = np.where(stats['y_ball_preds'][k] != stats['y_ball_preds'][k, 0])[0]
        stats['switch_ranks'].append(rks)

    # get confidences
    stats['confidences'] = np.max(stats['probs'], axis=2)

    # get index of first switch rank and the secondary label
    stats['first_switch_rank'] = -1 * np.ones(test_size, np.int32)
    stats['secondary_preds']   = -1 * np.ones(test_size, np.int32)  # the second label that was predicted for a noisier sample
    stats['no_sw_pred_inds'] = []
    for k in range(test_size):
        if stats['switch_ranks'][k].size != 0:
            stats['first_switch_rank'][k] = stats['switch_ranks'][k][0]
            stats['secondary_preds'][k] = stats['y_ball_preds'][k, stats['first_switch_rank'][k]]
        else:
            stats['no_sw_pred_inds'].append(k)

    if len(stats['no_sw_pred_inds']) > 0:
        print('These images have no pred switch: {}'.format(stats['no_sw_pred_inds']))

    stats['confidences_prime']     = -1 * np.ones((test_size, num_points))  # for the first label that was predicted for a noisier sample
    stats['confidences_secondary'] = -1 * np.ones((test_size, num_points))  # for the second label that was predicted for a noisier sample
    for k in range(test_size):
        stats['confidences_prime'][k] = stats['probs'][k, :, stats['y_ball_preds'][k, 0]]
        if k not in stats['no_sw_pred_inds']:
            stats['confidences_secondary'][k] = stats['probs'][k, :, stats['secondary_preds'][k]]

    # get pil mat
    stats['pil_mat'] = np.zeros((test_size, num_points, num_classes, num_classes))
    for cls in range(num_classes):
        tmp_preds = stats['preds'].copy()
        tmp_preds[:, :, cls] = -np.inf
        stats['pil_mat'][:, :, cls] = scipy.special.softmax(tmp_preds, axis=2)
    stats['pil_mat_mean'] = stats['pil_mat'].mean(axis=1)  # mean over TTAs

def histogram_intersection(f1, f2, num_bins=100, range_limit=(-np.inf, np.inf)):
    min_edge = max(f1.min(), f2.min(), range_limit[0])
    max_edge = min(f1.max(), f2.max(), range_limit[1])
    bins = np.linspace(min_edge, max_edge + eps, num_bins + 1)
    h1 = np.histogram(f1, bins=bins)[0]
    h2 = np.histogram(f2, bins=bins)[0]
    assert len(h1) == len(h2)
    sm = 0
    for i in range(len(h1)):
        sm += min(h1[i], h2[i])
    return sm

def plot_hists(name, f1, f2, range_limit=(-np.inf, np.inf)):
    # plot both
    plt.figure()
    plt.hist(f1, alpha=0.5, label='normal', bins=100)
    plt.hist(f2, alpha=0.5, label='adv'   , bins=100)
    plt.legend(loc='upper right')
    plt.title(name)
    plt.show()

    # plot intersection only
    plt.figure()
    min_edge = max(f1.min(), f2.min(), range_limit[0])
    max_edge = min(f1.max(), f2.max(), range_limit[1])
    plt.hist(f1, alpha=0.5, label='normal', bins=100, range=[min_edge, max_edge + eps])
    plt.hist(f2, alpha=0.5, label='adv'   , bins=100, range=[min_edge, max_edge + eps])
    plt.legend(loc='upper right')
    plt.title(name + ' (intersection)')
    plt.show()

def search_for_best_rank(f1, f2, num_bins=100, range_limit=(-np.inf, np.inf)):
    num_points = f1.shape[1]
    best_separation = np.inf  # lower is better
    best_top_rank = num_points

    for top_rank in range(50, num_points + 1, 50):
        top_rank -= 1
        f1_sel = f1[:, top_rank]
        f2_sel = f2[:, top_rank]
        # separation = search_func(f1_sel, f2_sel, num_bins, range_limit)
        separation = histogram_intersection(f1_sel, f2_sel, num_bins, range_limit)
        print('top_rank {}: separation={}'.format(top_rank + 1, separation))  # debug
        if separation <= best_separation:
            best_separation = separation
            best_top_rank = top_rank

    return best_top_rank

def search_for_best_thd(f1, f2, search_func=histogram_intersection):
    num_thds = f1.shape[1]
    best_separation = np.inf
    best_thd_pos = num_thds

    for thd_pos in range(num_thds):
        f1_sel = f1[:, thd_pos]
        f2_sel = f2[:, thd_pos]
        separation = search_func(f1_sel, f2_sel)
        print('thd_pos {}: separation={}'.format(thd_pos, separation))  # debug
        if separation <= best_separation:
            best_separation = separation
            best_thd_pos = thd_pos

    return best_thd_pos

def register_rank_feature(base_name, f1, f2, inds, rank=None, num_bins=100, range_limit=(-np.inf, np.inf), plot=PLOT):
    if rank is not None:
        best_top_rank = rank - 1
    else:
        best_top_rank = search_for_best_rank(f1[inds], f2[inds], num_bins, range_limit)
    feature_name = base_name + '_up_to_rank_{}'.format(best_top_rank + 1)
    if plot:
        plot_hists(feature_name, f1[inds, best_top_rank], f2[inds, best_top_rank], range_limit)
    return feature_name, f1[:, best_top_rank], f2[:, best_top_rank]

def register_thd_feature(base_name, f1, f2, inds, thds, thd=None, range_limit=(-np.inf, np.inf), plot=PLOT):
    if thd is not None:
        best_thd_pos = thds.index(thd)
        top_thd = thd
    else:
        best_thd_pos = search_for_best_thd(f1[inds], f2[inds])
        top_thd = thds[best_thd_pos]
    feature_name = base_name + '_w_thd_{}'.format(top_thd)
    if plot:
        plot_hists(feature_name, f1[inds, best_thd_pos], f2[inds, best_thd_pos], range_limit)
    return feature_name, f1[:, best_thd_pos], f2[:, best_thd_pos]

def register_common_feature(base_name, f1, f2, inds, range_limit=(-np.inf, np.inf), plot=PLOT):
    feature_name = base_name
    if plot:
        plot_hists(feature_name, f1[inds], f2[inds], range_limit)
    return feature_name, f1, f2

@to_features
def register_intg_loss(dataset, stats, stats_adv, inds):
    """
    Feature: integral loss up until a certain top rank
    :return: top_rank, normal features, and adv features
    """
    # name = inspect.stack()[0][3].split('register_')[1]
    name = 'intg_loss'
    f1 = np.cumsum(stats['losses']    , axis=1)
    f2 = np.cumsum(stats_adv['losses'], axis=1)
    return register_rank_feature(name, f1, f2, inds, FEATURES_RANKS[name][dataset])

@to_features
def register_intg_rel_loss(dataset, stats, stats_adv, inds):
    """
    Feature: integral loss up until a certain top rank
    :return: top_rank, normal features, and adv features
    """
    name = 'intg_rel_loss'
    f1 = np.cumsum(stats['rel_losses']    , axis=1)
    f2 = np.cumsum(stats_adv['rel_losses'], axis=1)
    return register_rank_feature(name, f1, f2, inds, FEATURES_RANKS[name][dataset], num_bins=1000, range_limit=(-np.inf, 2e5))

@to_features
def register_max_rel_loss(dataset, stats, stats_adv, inds):
    """
    Feature: integral loss up until a certain top rank
    :return: top_rank, normal features, and adv features
    """
    name = 'max_rel_loss'
    num_points = stats['preds'].shape[1]
    f1 = np.zeros_like(stats['rel_losses'])
    f2 = np.zeros_like(stats_adv['rel_losses'])
    for j in range(num_points):
        f1[:, j] = stats['rel_losses'][:, 0:j+1].max(axis=1)
        f2[:, j] = stats_adv['rel_losses'][:, 0:j+1].max(axis=1)
    return register_rank_feature(name, f1, f2, inds, FEATURES_RANKS[name][dataset])

@to_features
def register_rank_at_thd_rel_loss(dataset, stats, stats_adv, inds):
    name = 'rank_at_thd_rel_loss'
    thds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    test_size = stats['preds'].shape[0]
    f1 = np.zeros((test_size, len(thds)))
    f2 = np.zeros((test_size, len(thds)))

    rel_losses = stats['rel_losses']  # shape = (test_size, num_points)
    max_vals = rel_losses.max(axis=1)  # shape = test_size
    for t in range(len(thds)):
        thd = thds[t]
        thd_vals = thd * max_vals  # shape = test_size
        for k in range(test_size):
            f1[k, t] = np.argmax(rel_losses[k] > thd_vals[k])

    rel_losses = stats_adv['rel_losses']  # shape = (test_size, num_points)
    max_vals = rel_losses.max(axis=1)  # shape = test_size
    for t in range(len(thds)):
        thd = thds[t]
        thd_vals = thd * max_vals  # shape = test_size
        for k in range(test_size):
            f2[k, t] = np.argmax(rel_losses[k] > thd_vals[k])

    return register_thd_feature(name, f1, f2, inds, thds, FEATURES_RANKS[name][dataset])

@to_features
def register_rank_at_first_pred_switch(dataset, stats, stats_adv, inds):
    name = 'rank_at_first_pred_switch'
    test_size, num_points = stats['preds'].shape[0:2]
    f1 = -1 * np.ones(test_size, dtype=np.int32)
    f2 = -1 * np.ones(test_size, dtype=np.int32)

    for k in range(test_size):
        if stats['switch_ranks'][k].size != 0:
            f1[k] = stats['switch_ranks'][k][0]
        if stats_adv['switch_ranks'][k].size != 0:
            f2[k] = stats_adv['switch_ranks'][k][0]

    # replace value of -1 with 2 * num_points
    f1 = np.where(f1 == -1, 2 * num_points, f1)

    unswitched_adv_inds = np.where(f2 == -1)[0]
    if unswitched_adv_inds.size != 0:
        print('WARNING: these adversarial indices are never switched:\n{}'.format(unswitched_adv_inds))
    f2 = np.where(f2 == -1, 2 * num_points, f2)
    return register_common_feature(name, f1, f2, inds)

@to_features
def register_num_pred_switches(dataset, stats, stats_adv, inds):
    """
    Feature: number of prediction switches until a specific rank
    :return: top_rank, normal features, and adv features
    """
    name = 'num_pred_switches'
    test_size, num_points = stats['preds'].shape[0:2]
    f1 = np.zeros((test_size, num_points), dtype=np.int32)
    f2 = np.zeros((test_size, num_points), dtype=np.int32)
    for k in range(test_size):
        for sw_rnk in stats['switch_ranks'][k]:
            f1[k, sw_rnk:] += 1
        for sw_rnk in stats_adv['switch_ranks'][k]:
            f2[k, sw_rnk:] += 1
    return register_rank_feature(name, f1, f2, inds, FEATURES_RANKS[name][dataset])

@to_features
def register_mean_loss_for_initial_label(dataset, stats, stats_adv, inds):
    name = 'mean_loss_for_initial_label'
    test_size, num_points = stats['preds'].shape[0:2]
    f1 = np.zeros(test_size)
    f2 = np.zeros(test_size)

    for k in range(test_size):
        cnt = 0
        for j in range(num_points):
            if j not in stats['switch_ranks'][k]:
                cnt += 1
                f1[k] += stats['losses'][k, j]
        if cnt > 0:
            f1[k] /= cnt

        cnt = 0
        for j in range(num_points):
            if j not in stats_adv['switch_ranks'][k]:
                cnt += 1
                f2[k] += stats_adv['losses'][k, j]
        if cnt > 0:
            f2[k] /= cnt
    return register_common_feature(name, f1, f2, inds)

@to_features
def register_mean_rel_loss_for_initial_label(dataset, stats, stats_adv, inds):
    name = 'mean_rel_loss_for_initial_label'
    test_size, num_points = stats['preds'].shape[0:2]
    f1 = np.zeros(test_size)
    f2 = np.zeros(test_size)

    for k in range(test_size):
        cnt = 0
        for j in range(num_points):
            if j not in stats['switch_ranks'][k]:
                cnt += 1
                f1[k] += stats['rel_losses'][k, j]
        if cnt > 0:
            f1[k] /= cnt

        cnt = 0
        for j in range(num_points):
            if j not in stats_adv['switch_ranks'][k]:
                cnt += 1
                f2[k] += stats_adv['rel_losses'][k, j]
        if cnt > 0:
            f2[k] /= cnt
    return register_common_feature(name, f1, f2, inds)

@to_features
def register_intg_confidences_prime(dataset, stats, stats_adv, inds):
    name = 'intg_confidences_prime'
    f1 = np.cumsum(stats['confidences'], axis=1)
    f2 = np.cumsum(stats_adv['confidences'], axis=1)
    return register_rank_feature(name, f1, f2, inds, FEATURES_RANKS[name][dataset])

@to_features
def register_intg_confidences_prime_specific(dataset, stats, stats_adv, inds):
    name = 'intg_confidences_prime_specific'
    f1 = stats['confidences_prime']
    f2 = stats_adv['confidences_prime']

    f1 = np.cumsum(f1, axis=1)
    f2 = np.cumsum(f2, axis=1)
    return register_rank_feature(name, f1, f2, inds, FEATURES_RANKS[name][dataset])

@to_features
def register_intg_confidences_secondary(dataset, stats, stats_adv, inds):
    name = 'intg_confidences_secondary'
    test_size, num_points = stats['preds'].shape[0:2]
    f1 = np.zeros((test_size, num_points))
    f2 = np.zeros((test_size, num_points))
    for k in range(test_size):
        for j in range(num_points):
            f1[k, j] = np.sort(stats['probs'][k, j])[-2]
            f2[k, j] = np.sort(stats_adv['probs'][k, j])[-2]

    f1 = np.cumsum(f1, axis=1)
    f2 = np.cumsum(f2, axis=1)
    return register_rank_feature(name, f1, f2, inds, FEATURES_RANKS[name][dataset])

@to_features
def register_intg_confidences_secondary_specific(dataset, stats, stats_adv, inds):
    name = 'intg_confidences_secondary_specific'
    f1 = stats['confidences_secondary']
    f2 = stats_adv['confidences_secondary']

    f1 = np.cumsum(f1, axis=1)
    f2 = np.cumsum(f2, axis=1)
    f1[stats['no_sw_pred_inds']] = 0.0
    f2[stats_adv['no_sw_pred_inds']] = 0.0
    assert (f1 >= 0.0).all()
    assert (f2 >= 0.0).all()
    return register_rank_feature(name, f1, f2, inds, FEATURES_RANKS[name][dataset])

@to_features
def register_intg_delta_confidences_prime_rest(dataset, stats, stats_adv, inds):
    name = 'intg_delta_confidences_prime_rest'
    test_size, num_points = stats['preds'].shape[0:2]
    f1 = np.empty((test_size, num_points))
    f2 = np.empty((test_size, num_points))
    for k in range(test_size):
        for j in range(num_points):
            probs_sorted = np.sort(stats['probs'][k, j])
            rest, first = probs_sorted[:-1], probs_sorted[-1]
            f1[k, j] = max(0, first - np.sum(rest))

            probs_sorted = np.sort(stats_adv['probs'][k, j])
            rest, first = probs_sorted[:-1], probs_sorted[-1]
            f2[k, j] = max(0, first - np.sum(rest))

    f1 = np.cumsum(f1, axis=1)
    f2 = np.cumsum(f2, axis=1)
    return register_rank_feature(name, f1, f2, inds, FEATURES_RANKS[name][dataset])

@to_features
def register_intg_delta_confidences_prime_secondary_specific(dataset, stats, stats_adv, inds):
    name = 'intg_delta_confidences_prime_secondary_specific'
    test_size, num_points = stats['preds'].shape[0:2]
    f1 = -1 * np.ones((test_size, num_points))
    f2 = -1 * np.ones((test_size, num_points))
    for k in range(test_size):
        if k not in stats['no_sw_pred_inds']:
            f1[k] = stats['confidences_prime'][k] - stats['confidences_secondary'][k]
        else:
            f1[k] = stats['confidences_prime'][k]
        if k not in stats_adv['no_sw_pred_inds']:
            f2[k] = stats_adv['confidences_prime'][k] - stats_adv['confidences_secondary'][k]
        else:
            f2[k] = stats_adv['confidences_prime'][k]

    f1 = np.cumsum(f1, axis=1)
    f2 = np.cumsum(f2, axis=1)

    return register_rank_feature(name, f1, f2, inds, FEATURES_RANKS[name][dataset])

@to_features
def register_delta_probs_prime_secondary_excl_rest(dataset, stats, stats_adv, inds):
    name = 'delta_probs_prime_secondary_excl_rest'
    test_size, num_points, num_classes = stats['preds'].shape
    probs_first_second     = np.zeros((test_size, 2))
    probs_first_second_adv = np.zeros((test_size, 2))
    for k in range(test_size):
        first_cls, second_cls = stats['probs'][k, 0].argsort()[[-1, -2]]
        probs_tmp = stats['probs'][k].copy()
        for j in range(num_classes):
            if j not in [first_cls, second_cls]:
                probs_tmp[:, j] = -np.inf
        probs_new = scipy.special.softmax(probs_tmp, axis=1)
        probs_first_second[k, 0] = probs_new[:, first_cls].mean()
        probs_first_second[k, 1] = probs_new[:, second_cls].mean()

        first_cls, second_cls = stats_adv['probs'][k, 0].argsort()[[-1, -2]]
        probs_tmp = stats_adv['probs'][k].copy()
        for j in range(num_classes):
            if j not in [first_cls, second_cls]:
                probs_tmp[:, j] = -np.inf
        probs_new = scipy.special.softmax(probs_tmp, axis=1)
        probs_first_second_adv[k, 0] = probs_new[:, first_cls].mean()
        probs_first_second_adv[k, 1] = probs_new[:, second_cls].mean()

    f1 = probs_first_second[:, 0]     - probs_first_second[:, 1]
    f2 = probs_first_second_adv[:, 0] - probs_first_second_adv[:, 1]

    return register_common_feature(name, f1, f2, inds, FEATURES_RANKS[name][dataset])

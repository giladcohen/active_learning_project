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

rand_gen = np.random.RandomState(12345)
eps = 1e-10

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


def plot_ttas(x, x_adv, args, f2_inds, n_imgs=5, n_dist=10):
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
    inds = rand_gen.choice([si for si in f2_inds if si < 500], n_imgs, replace=False)
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
    test_size = len(stats['preds'])

    # get softmax probabilities
    stats['probs'] = scipy.special.softmax(stats['preds'], axis=2)

    # get relative losses
    losses = stats['losses']
    rel_losses = np.zeros_like(losses)
    for k in range(test_size):
        rel_losses[k] = (losses[k] - losses[k, 0]) / losses[k, 0]
    stats['rel_losses'] = rel_losses

    # get hard predictions
    stats['y_ball_preds'] = stats['probs'].argmax(axis=2)

    # get all tta ranks (orders) that yielded different prediction than the original image
    switch_ranks = []
    for k in range(test_size):
        rks = np.where(stats['y_ball_preds'][k] != stats['y_ball_preds'][k, 0])[0]
        switch_ranks.append(rks)
    stats['switch_ranks'] = switch_ranks

    # get confidences
    stats['confidences'] = np.max(stats['probs'], axis=2)

    # get images index without any switched pred:
    no_sw_pred_inds = []
    for k in range(test_size):
        if switch_ranks[k].size == 0:
            print('image i={} has no pred switch'.format(k))
            no_sw_pred_inds.append(k)
    stats['no_sw_pred_inds'] = no_sw_pred_inds

def histogram_intersection(h1, h2):
    assert len(h1) == len(h2)
    sm = 0
    for i in range(len(h1)):
        sm += min(h1[i], h2[i])
    return sm

def plot_hists(name, f1, f2):
    # plot both
    plt.figure()
    plt.hist(f1, alpha=0.5, label='normal', bins=100)
    plt.hist(f2, alpha=0.5, label='adv'   , bins=100)
    plt.legend(loc='upper right')
    plt.title(name)
    plt.show()

    # plot intersection only
    plt.figure()
    min_edge = max(f1.min(), f2.min())
    max_edge = min(f1.max(), f2.max())
    plt.hist(f1, alpha=0.5, label='normal', bins=100, range=[min_edge, max_edge + eps])
    plt.hist(f2, alpha=0.5, label='adv'   , bins=100, range=[min_edge, max_edge + eps])
    plt.legend(loc='upper right')
    plt.title(name + ' (intersection)')
    plt.show()

def search_for_best_rank(f1, f2):
    num_points = f1.shape[1]
    best_intersection = np.inf
    best_top_rank = num_points

    for top_rank in range(50, num_points + 1, 50):
        top_rank -= 1
        min_edge = max(f1[:, top_rank].min(), f2[:, top_rank].min())
        max_edge = min(f1[:, top_rank].max(), f2[:, top_rank].max())
        bins = np.linspace(min_edge, max_edge + eps, 101)
        hist1 = np.histogram(f1[:, top_rank], bins=bins)
        hist2 = np.histogram(f2[:, top_rank], bins=bins)
        intersection = histogram_intersection(hist1[0], hist2[0])
        print('top_rank {}: intersection={}'.format(top_rank + 1, intersection))  # debug
        if intersection <= best_intersection:
            best_intersection = intersection
            best_top_rank = top_rank

    return best_top_rank

def search_for_best_thd(f1, f2):
    num_thds = f1.shape[1]
    best_intersection = np.inf
    best_thd_pos = num_thds

    for thd_pos in range(num_thds):
        min_edge = max(f1[:, thd_pos].min(), f2[:, thd_pos].min())
        max_edge = min(f1[:, thd_pos].max(), f2[:, thd_pos].max())
        bins = np.linspace(min_edge, max_edge + eps, 101)
        hist1 = np.histogram(f1[:, thd_pos], bins=bins)
        hist2 = np.histogram(f2[:, thd_pos], bins=bins)
        intersection = histogram_intersection(hist1[0], hist2[0])
        print('thd_pos {}: intersection={}'.format(thd_pos, intersection))  # debug
        if intersection <= best_intersection:
            best_intersection = intersection
            best_thd_pos = thd_pos

    return best_thd_pos

def register_rank_feature(base_name, f1, f2, inds, plot=True):
    best_top_rank = search_for_best_rank(f1[inds], f2[inds])
    feature_name = base_name + '_up_to_rank_{}'.format(best_top_rank + 1)
    if plot:
        plot_hists(feature_name, f1[inds, best_top_rank], f2[inds, best_top_rank])
    return feature_name, f1[:, best_top_rank], f2[:, best_top_rank]

def register_thd_feature(base_name, f1, f2, inds, thds, plot=True):
    best_thd_pos = search_for_best_thd(f1[inds], f2[inds])
    top_thd = thds[best_thd_pos]
    feature_name = base_name + '_up_to_thd_{}'.format(top_thd)
    if plot:
        plot_hists(feature_name, f1[inds, best_thd_pos], f2[inds, best_thd_pos])
    return feature_name, f1[:, best_thd_pos], f2[:, best_thd_pos]

def register_common_feature(base_name, f1, f2, inds, plot=True):
    feature_name = base_name
    if plot:
        plot_hists(feature_name, f1[inds], f2[inds])
    return feature_name, f1, f2

@to_features
def register_intg_loss(stats, stats_adv, inds):
    """
    Feature: integral loss up until a certain top rank
    :return: top_rank, normal features, and adv features
    """

    f1 = np.cumsum(stats['losses']    , axis=1)
    f2 = np.cumsum(stats_adv['losses'], axis=1)
    return register_rank_feature('intg_loss', f1, f2, inds)

@to_features
def register_intg_rel_loss(stats, stats_adv, inds):
    """
    Feature: integral loss up until a certain top rank
    :return: top_rank, normal features, and adv features
    """

    f1 = np.cumsum(stats['rel_losses']    , axis=1)
    f2 = np.cumsum(stats_adv['rel_losses'], axis=1)
    return register_rank_feature('intg_rel_loss', f1, f2, inds)

@to_features
def register_max_rel_loss(stats, stats_adv, inds):
    """
    Feature: integral loss up until a certain top rank
    :return: top_rank, normal features, and adv features
    """
    num_points = stats['preds'].shape[1]
    f1 = np.zeros_like(stats['rel_losses'])
    f2 = np.zeros_like(stats_adv['rel_losses'])
    for j in range(num_points):
        f1[:, j] = stats['rel_losses'][:, 0:j+1].max(axis=1)
        f2[:, j] = stats_adv['rel_losses'][:, 0:j+1].max(axis=1)

    return register_rank_feature('max_relative_loss', f1, f2, inds)

@to_features
def register_rank_at_thd_rel_loss(stats, stats_adv, inds):
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

    return register_thd_feature('max_relative_loss', f1, f2, inds, thds)

@to_features
def register_rank_at_first_pred_switch(stats, stats_adv, inds):
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
    return register_common_feature('rank_at_first_pred_switch', f1, f2, inds)

@to_features
def register_num_pred_switches(stats, stats_adv, inds):
    """
    Feature: number of prediction switches until a specific rank
    :return: top_rank, normal features, and adv features
    """
    test_size, num_points = stats['preds'].shape[0:2]
    f1 = np.zeros((test_size, num_points), dtype=np.int32)
    f2 = np.zeros((test_size, num_points), dtype=np.int32)
    for k in range(test_size):
        for sw_rnk in stats['switch_ranks'][k]:
            f1[k, sw_rnk:] += 1
        for sw_rnk in stats_adv['switch_ranks'][k]:
            f2[k, sw_rnk:] += 1
    return register_rank_feature('num_pred_switches', f1, f2, inds)

@to_features
def register_mean_loss_for_initial_label(stats, stats_adv, inds):
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

    return register_common_feature('mean_loss_for_initial_label', f1, f2, inds)

@to_features
def register_mean_rel_loss_for_initial_label(stats, stats_adv, inds):
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

    return register_common_feature('mean_rel_loss_for_initial_label', f1, f2, inds)

@to_features
def register_sum_confidences(stats, stats_adv, inds):
    f1 = np.cumsum(stats['confidences'], axis=1)
    f2 = np.cumsum(stats_adv['confidences'], axis=1)
    return register_rank_feature('sum_confidences', f1, f2, inds)









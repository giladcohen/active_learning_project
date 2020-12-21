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

rand_gen = np.random.RandomState(12345)

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
    plt.figure()
    plt.hist(f1, alpha=0.5, label='normal', bins=100)
    plt.hist(f2, alpha=0.5, label='adv'   , bins=100)
    plt.legend(loc='upper right')
    plt.title(name)
    plt.show()

def search_for_best_rank(f1, f2):
    num_points = f1.shape[1]
    best_intersection = np.inf
    best_top_rank = num_points
    eps = 1e-10

    for top_rank in range(50, num_points + 1, 50):
        top_rank -= 1
        min_edge = min(f1[:, top_rank].min(), f2[:, top_rank].min())
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

def register_common_feature(base_name, f1, f2, plot=True):
    best_top_rank = search_for_best_rank(f1, f2)
    feature_name = base_name + '_up_to_rank_{}'.format(best_top_rank + 1)
    if plot:
        plot_hists(feature_name, f1[:, best_top_rank], f2[:, best_top_rank])
    return feature_name, f1[:, best_top_rank], f2[:, best_top_rank]

def register_intg_loss(stats, stats_adv, inds):
    """
    Feature: integral loss up until a certain top rank
    :return: top_rank, normal features, and adv features
    """

    f1 = np.cumsum(stats['losses'][inds]    , axis=1)
    f2 = np.cumsum(stats_adv['losses'][inds], axis=1)
    return register_common_feature('intg_loss', f1, f2)

def register_intg_rel_loss(stats, stats_adv, inds, plot=True):
    """
    Feature: integral loss up until a certain top rank
    :return: top_rank, normal features, and adv features
    """

    f1 = np.cumsum(stats['rel_losses'][inds]    , axis=1)
    f2 = np.cumsum(stats_adv['rel_losses'][inds], axis=1)
    return register_common_feature('intg_rel_loss', f1, f2)

def register_max_rel_loss(stats, stats_adv, inds, plot=True):
    """
    Feature: integral loss up until a certain top rank
    :return: top_rank, normal features, and adv features
    """
    num_points = stats['preds'].shape[1]
    f1 = np.zeros_like(stats['rel_losses'][inds])
    f2 = np.zeros_like(stats_adv['rel_losses'][inds])
    for j in range(num_points):
        f1[:, j] = stats['rel_losses'][inds, 0: j+1].max(axis=1)
        f2[:, j] = stats_adv['rel_losses'][inds, 0: j+1].max(axis=1)

    return register_common_feature('max_relative_loss', f1, f2)

def register_rank_at_thd_rel_loss(stats, stats_adv, inds):
    thd_options = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]













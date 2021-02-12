import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import numpy as np
import json
import os
import argparse
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import scipy
import numba as nb
from numba import njit
from tqdm import tqdm

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.utils import convert_tensor_to_image, calc_prob_wo_l, compute_roc, boolean_string

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch adversarial robustness testing')
parser.add_argument('--checkpoint_dir',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00',
                    type=str, help='dir containing the network checkpoint')
parser.add_argument('--src',
                    default='cw_targeted',
                    type=str, help='dir containing training features, relative to checkpoint_dir')
parser.add_argument('--dst',
                    default='cw_targeted',
                    type=str, help='dir containing testing features, relative to checkpoint_dir')
parser.add_argument('--defense',
                    default='mahalanobis',
                    type=str, help='name of defense')

# for LID:
parser.add_argument('--k_nearest', default=-1, type=int, help='number of nearest neighbors to use for LID/DkNN detection')

# for mahalanobis:
parser.add_argument('--magnitude', default=-1, type=float, help='number of nearest neighbors to use for LID/DkNN detection')

# for ours
parser.add_argument('--f_inds',
                    default='f1',
                    type=str, help='The type of images used for training features')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()
SRC_DIR  = os.path.join(args.checkpoint_dir, args.src, args.defense)
DST_DIR  = os.path.join(args.checkpoint_dir, args.dst, args.defense)
INDS_DIR = os.path.join(args.checkpoint_dir, args.src, 'inds')

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)

rand_gen = np.random.RandomState(12345)
# load inds
src_val_inds     = np.load(os.path.join(args.checkpoint_dir, args.src, 'inds', 'val_inds.npy'))
src_f0_inds_val  = np.load(os.path.join(args.checkpoint_dir, args.src, 'inds', 'f0_inds_val.npy'))
src_f1_inds_val  = np.load(os.path.join(args.checkpoint_dir, args.src, 'inds', 'f1_inds_val.npy'))
src_f2_inds_val  = np.load(os.path.join(args.checkpoint_dir, args.src, 'inds', 'f2_inds_val.npy'))
src_f3_inds_val  = np.load(os.path.join(args.checkpoint_dir, args.src, 'inds', 'f3_inds_val.npy'))
src_test_inds    = np.load(os.path.join(args.checkpoint_dir, args.src, 'inds', 'test_inds.npy'))
src_f0_inds_test = np.load(os.path.join(args.checkpoint_dir, args.src, 'inds', 'f0_inds_test.npy'))
src_f1_inds_test = np.load(os.path.join(args.checkpoint_dir, args.src, 'inds', 'f1_inds_test.npy'))
src_f2_inds_test = np.load(os.path.join(args.checkpoint_dir, args.src, 'inds', 'f2_inds_test.npy'))
src_f3_inds_test = np.load(os.path.join(args.checkpoint_dir, args.src, 'inds', 'f3_inds_test.npy'))

dst_val_inds     = np.load(os.path.join(args.checkpoint_dir, args.dst, 'inds', 'val_inds.npy'))
dst_f0_inds_val  = np.load(os.path.join(args.checkpoint_dir, args.dst, 'inds', 'f0_inds_val.npy'))
dst_f1_inds_val  = np.load(os.path.join(args.checkpoint_dir, args.dst, 'inds', 'f1_inds_val.npy'))
dst_f2_inds_val  = np.load(os.path.join(args.checkpoint_dir, args.dst, 'inds', 'f2_inds_val.npy'))
dst_f3_inds_val  = np.load(os.path.join(args.checkpoint_dir, args.dst, 'inds', 'f3_inds_val.npy'))
dst_test_inds    = np.load(os.path.join(args.checkpoint_dir, args.dst, 'inds', 'test_inds.npy'))
dst_f0_inds_test = np.load(os.path.join(args.checkpoint_dir, args.dst, 'inds', 'f0_inds_test.npy'))
dst_f1_inds_test = np.load(os.path.join(args.checkpoint_dir, args.dst, 'inds', 'f1_inds_test.npy'))
dst_f2_inds_test = np.load(os.path.join(args.checkpoint_dir, args.dst, 'inds', 'f2_inds_test.npy'))
dst_f3_inds_test = np.load(os.path.join(args.checkpoint_dir, args.dst, 'inds', 'f3_inds_test.npy'))

if args.f_inds == 'f0':
    src_inds_val = src_f0_inds_val
elif args.f_inds == 'f1':
    src_inds_val = src_f1_inds_val
elif args.f_inds == 'f2':
    src_inds_val = src_f2_inds_val
elif args.f_inds == 'f3':
    src_inds_val = src_f3_inds_val
else:
    raise AssertionError(args.f_inds + ' is not acceptable')

# metric functions:
def calc_adv_detection_metrics(detection_preds, detection_preds_adv):
    print('Calculating adv detection metrics...')
    acc_all = np.mean(detection_preds[dst_test_inds] == 0)
    acc_f1 = np.mean(detection_preds[dst_f1_inds_test] == 0)
    acc_f2 = np.mean(detection_preds[dst_f2_inds_test] == 0)
    acc_f3 = np.mean(detection_preds[dst_f3_inds_test] == 0)

    acc_all_adv = np.mean(detection_preds_adv[dst_test_inds] == 1)
    acc_f1_adv = np.mean(detection_preds_adv[dst_f1_inds_test] == 1)
    acc_f2_adv = np.mean(detection_preds_adv[dst_f2_inds_test] == 1)
    acc_f3_adv = np.mean(detection_preds_adv[dst_f3_inds_test] == 1)

    f1_test_preds_all = np.concatenate(
        (detection_preds_prob[dst_f1_inds_test], detection_preds_prob_adv[dst_f1_inds_test]), axis=0)
    f1_test_gt_all = np.concatenate((np.zeros(len(dst_f1_inds_test)), np.ones(len(dst_f1_inds_test))), axis=0)
    _, _, auc_score = compute_roc(f1_test_gt_all, f1_test_preds_all, plot=False)
    print('Adv detection Accuracy: all samples: {:.2f}/{:.2f}%. '
          'f1 samples: {:.2f}/{:.2f}%, f2 samples: {:.2f}/{:.2f}%, f3 samples: {:.2f}/{:.2f}%. '
          'AUC score: {:.5f}'
          .format(acc_all * 100, acc_all_adv * 100, acc_f1 * 100, acc_f1_adv * 100, acc_f2 * 100, acc_f2_adv * 100,
                  acc_f3 * 100, acc_f3_adv * 100, auc_score))

def calc_robust_metrics(robustness_preds, robustness_preds_adv):
    print('Calculating robustness metrics...')
    acc_all = np.mean(robustness_preds[dst_test_inds] == y_test[dst_test_inds])
    acc_f1 = np.mean(robustness_preds[dst_f1_inds_test] == y_test[dst_f1_inds_test])
    acc_f2 = np.mean(robustness_preds[dst_f2_inds_test] == y_test[dst_f2_inds_test])
    acc_f3 = np.mean(robustness_preds[dst_f3_inds_test] == y_test[dst_f3_inds_test])

    acc_all_adv = np.mean(robustness_preds_adv[dst_test_inds] == y_test[dst_test_inds])
    acc_f1_adv = np.mean(robustness_preds_adv[dst_f1_inds_test] == y_test[dst_f1_inds_test])
    acc_f2_adv = np.mean(robustness_preds_adv[dst_f2_inds_test] == y_test[dst_f2_inds_test])
    acc_f3_adv = np.mean(robustness_preds_adv[dst_f3_inds_test] == y_test[dst_f3_inds_test])

    print('Robust classification accuracy: all samples: {:.2f}/{:.2f}%, f1 samples: {:.2f}/{:.2f}%, f2 samples: {:.2f}/{:.2f}%, f3 samples: {:.2f}/{:.2f}%'
          .format(acc_all * 100, acc_all_adv * 100, acc_f1 * 100, acc_f1_adv * 100, acc_f2 * 100, acc_f2_adv * 100, acc_f3 * 100, acc_f3_adv * 100))

def load_characteristics(characteristics_file):
    X, Y = None, None
    data = np.load(characteristics_file)
    if X is None:
        X = data[:, :-1]
    if Y is None:
        Y = data[:, -1]  # labels only need to load once

    return X, Y

if args.defense in ['lid', 'mahalanobis']:
    train_characteristics_file_vec = []
    test_characteristics_file_vec = []
    if args.defense == 'lid':
        if args.k_nearest == -1:
            k_vec = np.arange(10, 41, 2)
        else:
            k_vec = [args.k_nearest]
        for k in k_vec:
            train_characteristics_file_vec.append(os.path.join(SRC_DIR, 'k_{}_train.npy'.format(k)))
            test_characteristics_file_vec.append(os.path.join(DST_DIR, 'k_{}_test.npy'.format(k)))
    else:
        if args.magnitude == -1:
            magnitude_vec = np.array([0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01])
        else:
            magnitude_vec = [args.magnitude]
        for mag in magnitude_vec:
            train_characteristics_file_vec.append(os.path.join(SRC_DIR, 'mag_{}_train.npy'.format(mag)))
            test_characteristics_file_vec.append(os.path.join(DST_DIR, 'mag_{}_test.npy'.format(mag)))

    for i in range(len(train_characteristics_file_vec)):
        train_characteristics_file = train_characteristics_file_vec[i]
        test_characteristics_file = test_characteristics_file_vec[i]
        print("Loading attacks...\nTraining file: {}\nTesting file: {}".format(train_characteristics_file, test_characteristics_file))
        X_train, Y_train = load_characteristics(train_characteristics_file)
        X_test, Y_test   = load_characteristics(test_characteristics_file)
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        #print("Train data size: ", X_train.shape)
        #print("Test data size: ", X_test.shape)

        ## Build detector
        lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
    
        ## Evaluate detector
        y_pred = lr.predict_proba(X_test)[:, 1]
        y_label_pred = lr.predict(X_test)

        _, _, auc_score = compute_roc(Y_test, y_pred, plot=True)
        precision = precision_score(Y_test, y_label_pred)
        recall = recall_score(Y_test, y_label_pred)
        acc = accuracy_score(Y_test, y_label_pred)
        print('Detector ROC-AUC score: {}, accuracy: {}, precision: {}, recall: {}'.format(auc_score, acc, precision, recall))
    exit(0)

# load train features:
features_index  = np.load(os.path.join(SRC_DIR, 'features_index_hist_by_{}.npy'.format(args.f_inds)))
normal_features = np.load(os.path.join(SRC_DIR, 'normal_features_hist_by_{}.npy'.format(args.f_inds)))
adv_features    = np.load(os.path.join(SRC_DIR, 'adv_features_hist_by_{}.npy'.format(args.f_inds)))
train_features = np.concatenate((normal_features[src_inds_val], adv_features[src_inds_val]))
train_labels   = np.concatenate((np.zeros(len(src_inds_val)), np.ones(len(src_inds_val))))

# load test features:
assert (features_index  == np.load(os.path.join(DST_DIR, 'features_index_hist_by_{}.npy'.format(args.f_inds)))).all()
assert (normal_features == np.load(os.path.join(DST_DIR, 'normal_features_hist_by_{}.npy'.format(args.f_inds)))).all()
test_normal_features = np.load(os.path.join(DST_DIR, 'normal_features_hist_by_{}.npy'.format(args.f_inds)))
test_adv_features    = np.load(os.path.join(DST_DIR, 'adv_features_hist_by_{}.npy'.format(args.f_inds)))

# fitting random forest classifier
clf = RandomForestClassifier(
    n_estimators=1000,
    criterion="entropy",  # gini or entropy
    max_depth=None, # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
                    # until all leaves contain less than min_samples_split samples.
    min_samples_split=10,
    min_samples_leaf=10,
    bootstrap=True, # Whether bootstrap samples are used when building trees.
                    # If False, the whole datset is used to build each tree.
    random_state=rand_gen,
    verbose=0,
    n_jobs=20
)
clf.fit(train_features, train_labels)
detection_preds          = clf.predict(test_normal_features)
detection_preds_adv      = clf.predict(test_adv_features)
detection_preds_prob     = clf.predict_proba(test_normal_features)[:, 1]
detection_preds_prob_adv = clf.predict_proba(test_adv_features)[:, 1]

calc_adv_detection_metrics(detection_preds, detection_preds_adv)

# now robustness...
y_test    = np.load(os.path.join(args.checkpoint_dir, 'normal', 'y_test.npy'))
preds     = np.load(os.path.join(args.checkpoint_dir, 'normal', args.defense, 'preds.npy'))
preds_adv = np.load(os.path.join(DST_DIR, 'preds_adv.npy'))
test_size, num_points, num_classes = preds.shape

def get_mini_stats(preds):
    stats = {}
    stats['preds'] = preds
    stats['probs'] = scipy.special.softmax(stats['preds'], axis=2)
    stats['y_ball_preds'] = stats['probs'].argmax(axis=2)

    if num_classes <= 10:  # use the fast way:
        stats['pil_mat'] = np.zeros((test_size, num_points, num_classes, num_classes), np.float32)
        for cls in range(num_classes):
            tmp_preds = stats['preds'].copy()
            tmp_preds[:, :, cls] = -np.inf
            stats['pil_mat'][:, :, cls] = scipy.special.softmax(tmp_preds, axis=2)
        stats['pil_mat_mean'] = stats['pil_mat'].mean(axis=1)  # mean over TTAs
    else:  # use the slow way (due to high memory)
        stats['pil_mat_mean'] = np.zeros((test_size, num_classes, num_classes), np.float32)
        for k in range(test_size):
            for cls in range(num_classes):
                pil_tmp = stats['preds'][k].copy()
                pil_tmp[:, cls] = -np.inf
                pil_tmp = scipy.special.softmax(pil_tmp, axis=1)
                stats['pil_mat_mean'][k, cls] = pil_tmp.mean(axis=0)
    return stats

stats     = get_mini_stats(preds)
stats_adv = get_mini_stats(preds_adv)

robustness_probs     = np.empty((test_size, num_classes))
robustness_probs_adv = np.empty((test_size, num_classes))
for k in range(test_size):
    l = stats['y_ball_preds'][k, 0]
    p_is_adv = detection_preds_prob[k]
    p_vec_norm = stats['probs'][k, 0]
    p_vec_adv = calc_prob_wo_l(stats['preds'][k, 0], l=l)
    robustness_probs[k] = p_vec_adv * p_is_adv + p_vec_norm * (1 - p_is_adv)

    l = stats_adv['y_ball_preds'][k, 0]
    p_is_adv = detection_preds_prob_adv[k]
    p_vec_norm = stats_adv['probs'][k, 0]
    p_vec_adv = calc_prob_wo_l(stats_adv['preds'][k, 0], l=l)
    robustness_probs_adv[k] = p_vec_adv * p_is_adv + p_vec_norm * (1 - p_is_adv)

robustness_preds     = robustness_probs.argmax(axis=1)
robustness_preds_adv = robustness_probs_adv.argmax(axis=1)
calc_robust_metrics(robustness_preds, robustness_preds_adv)

robustness_probs     = np.empty((test_size, num_classes))
robustness_probs_adv = np.empty((test_size, num_classes))
for k in range(test_size):
    l = stats['y_ball_preds'][k, 0]
    p_is_adv = detection_preds_prob[k]
    p_vec_norm = stats['probs'][k].mean(axis=0)
    p_vec_adv = stats['pil_mat_mean'][k, l]
    robustness_probs[k] = p_vec_adv * p_is_adv + p_vec_norm * (1 - p_is_adv)

    l = stats_adv['y_ball_preds'][k, 0]
    p_is_adv = detection_preds_prob_adv[k]
    p_vec_norm = stats_adv['probs'][k].mean(axis=0)
    p_vec_adv = stats_adv['pil_mat_mean'][k, l]
    robustness_probs_adv[k] = p_vec_adv * p_is_adv + p_vec_norm * (1 - p_is_adv)

robustness_preds     = robustness_probs.argmax(axis=1)
robustness_preds_adv = robustness_probs_adv.argmax(axis=1)
calc_robust_metrics(robustness_preds, robustness_preds_adv)

# calculate a third robustness classification, if the image is normal, use only the original sample,
# and if the image is adversarial, use the
robustness_probs     = np.empty((test_size, num_classes))
robustness_probs_adv = np.empty((test_size, num_classes))
for k in range(test_size):
    l = stats['y_ball_preds'][k, 0]
    p_is_adv = detection_preds_prob[k]
    if p_is_adv <= 0.5:  # probably normal
        p_vec_norm = stats['probs'][k, 0]
        p_vec_adv = calc_prob_wo_l(stats['preds'][k, 0], l=l)
    else:  # probably adv
        p_vec_norm = stats['probs'][k].mean(axis=0)
        p_vec_adv = stats['pil_mat_mean'][k, l]
    robustness_probs[k] = p_vec_adv * p_is_adv + p_vec_norm * (1 - p_is_adv)

    l = stats_adv['y_ball_preds'][k, 0]
    p_is_adv = detection_preds_prob_adv[k]
    if p_is_adv <= 0.5:  # probably normal
        p_vec_norm = stats_adv['probs'][k, 0]
        p_vec_adv = calc_prob_wo_l(stats_adv['preds'][k, 0], l=l)
    else:  # probably adv
        p_vec_norm = stats_adv['probs'][k].mean(axis=0)
        p_vec_adv = stats_adv['pil_mat_mean'][k, l]
    robustness_probs_adv[k] = p_vec_adv * p_is_adv + p_vec_norm * (1 - p_is_adv)

robustness_preds     = robustness_probs.argmax(axis=1)
robustness_preds_adv = robustness_probs_adv.argmax(axis=1)
calc_robust_metrics(robustness_preds, robustness_preds_adv)



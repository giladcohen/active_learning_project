import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary

from tqdm import tqdm
import numpy as np
import json
import os
import argparse
import sys
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.utils import boolean_string, majority_vote, get_ensemble_paths, add_feature
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor
from scipy.special import softmax
from cleverhans.utils import to_categorical, batch_indices

import matplotlib.pyplot as plt
from art.classifiers import PyTorchClassifier
# from active_learning_project.classifiers.pytorch_ext_classifier import PyTorchExtClassifier

parser = argparse.ArgumentParser(description='Adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/Users/giladcohen/data/gilad/logs/adv_robustness/svhn/resnet34/adv_robust/robust_resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='ead', type=str, help='attack directory')
parser.add_argument('--rev_dir', default='', type=str, help='reverse dir')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--net_pool', default='main', type=str, help='networks pool: main, ensemble, all')
parser.add_argument('--img_pool', default='orig', type=str, help='images pool: orig, rev, all')
parser.add_argument('--method', default='simple', type=str, help='method of defense: simple, inference_svm')
parser.add_argument('--train_on', default='adv', type=str, help='normal, adv, all')
parser.add_argument('--temperature', default=1, type=float, help='normal, adv')
parser.add_argument('--pca_dims', default=-1, type=int, help='if not -1, apply PCA to svm with dims')
parser.add_argument('--subset', default=-1, type=int, help='attack only subset of test set')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)

CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

if args.rev_dir != '':
    REV_DIR = os.path.join(ATTACK_DIR, args.rev_dir)
    with open(os.path.join(REV_DIR, 'defense_args.txt'), 'r') as f:
        defense_args = json.load(f)
else:
    REV_DIR = None
    defense_args = None

ENSEMBLE_DIR = os.path.dirname(args.checkpoint_dir)
ENSEMBLE_DIR_DUMP = os.path.join(ATTACK_DIR, 'ensemble')

batch_size = args.batch_size
T = args.temperature  # Temperature for softmax
rand_gen = np.random.RandomState(seed=12345)

# Data
print('==> Preparing data..')
testloader = get_test_loader(
    dataset=train_args['dataset'],
    batch_size=batch_size,
    num_workers=1,
    pin_memory=True
)

global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
train_inds = np.asarray(global_state['train_inds'])
val_inds = np.asarray(global_state['val_inds'])
classes = testloader.dataset.classes
test_size = len(testloader.dataset)
test_inds = np.arange(test_size)

X_test           = get_normalized_tensor(testloader, batch_size)
y_test           = np.asarray(testloader.dataset.targets)
y_test_preds     = np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy'))

X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
y_test_adv_preds = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))
if targeted:
    y_test_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))

# Model
print('==> Building model..')
if train_args['net'] == 'resnet34':
    net = ResNet34(num_classes=len(classes), activation=train_args['activation'])
elif train_args['net'] == 'resnet101':
    net = ResNet101(num_classes=len(classes), activation=train_args['activation'])
else:
    raise AssertionError("network {} is unknown".format(train_args['net']))
net = net.to(device)

# summary(net, (3, 32, 32))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net.load_state_dict(global_state['best_net'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=train_args['lr'],
    momentum=train_args['mom'],
    weight_decay=0.0,  # train_args['wd'],
    nesterov=train_args['mom'] > 0)

# get and assert preds:
net.eval()
classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=criterion,
                               optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(classes))

# load all calculated features:
load_main     = args.net_pool in ['main', 'all']
load_ensemble = args.net_pool in ['ensemble', 'all']
load_orig     = args.img_pool in ['orig', 'all']
load_rev      = args.img_pool in ['rev', 'all']
train_normal  = args.train_on in ['normal', 'all']
train_adv     = args.train_on in ['adv', 'all']

# main
y_main_logits         = np.load(os.path.join(ATTACK_DIR, 'y_test_logits.npy'))
y_adv_main_logits     = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_logits.npy'))

# ensemble
y_net_logits          = np.load(os.path.join(ENSEMBLE_DIR_DUMP, 'y_test_net_logits_mat.npy'))
y_adv_net_logits      = np.load(os.path.join(ENSEMBLE_DIR_DUMP, 'y_test_adv_net_logits_mat.npy'))

subset = args.subset  # must be explicit!
if subset != -1:  # because of debug defense
    X_test = X_test[:subset]
    y_test = y_test[:subset]
    y_test_preds = y_test_preds[:subset]
    X_test_adv = X_test_adv[:subset]
    y_test_adv_preds = y_test_adv_preds[:subset]
    if targeted:
        y_test_adv = y_test_adv[:subset]

    y_main_logits = y_main_logits[:subset]
    y_adv_main_logits = y_adv_main_logits[:subset]
    y_net_logits = y_net_logits[:subset]
    y_adv_net_logits = y_adv_net_logits[:subset]

    test_size = subset

num_batches = int(np.ceil(test_size/batch_size))

# cross
y_cross_logits        = np.concatenate((np.expand_dims(y_main_logits, axis=1), y_net_logits), axis=1)          # (N, 10, #class)
y_adv_cross_logits    = np.concatenate((np.expand_dims(y_adv_main_logits, axis=1), y_adv_net_logits), axis=1)  # (N, 10, #class)

# calculating preds:
y_cross_preds         = softmax((1/T) * y_cross_logits, axis=2)          # (N, 10, #cls)
y_adv_cross_preds     = softmax((1/T) * y_adv_cross_logits, axis=2)      # (N, 10, #cls)

# what are the samples we care about? net_succ (not attack_succ. it is irrelevant)
# load inds
val_inds     = np.load(os.path.join(ATTACK_DIR, 'inds', 'val_inds.npy'))
f0_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f0_inds_val.npy'))
f1_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f1_inds_val.npy'))
f2_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f2_inds_val.npy'))
f3_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f3_inds_val.npy'))

test_inds    = np.load(os.path.join(ATTACK_DIR, 'inds', 'test_inds.npy'))
f0_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f0_inds_test.npy'))
f1_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f1_inds_test.npy'))
f2_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f2_inds_test.npy'))
f3_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f3_inds_test.npy'))

print("Number of test samples: {}. #net_succ: {}. #net_succ_attack_flip: {}. #net_succ_attack_succ: {}"
      .format(len(test_inds), len(f1_inds_test), len(f2_inds_test), len(f3_inds_test)))

if load_rev:
    assert test_size == defense_args.get('subset')
    X_test_rev = np.load(os.path.join(REV_DIR, 'X_test_rev.npy'))
    X_test_rev_mat = np.load(os.path.join(REV_DIR, 'X_test_rev_mat.npy'))
    X_test_adv_rev = np.load(os.path.join(REV_DIR, 'X_test_adv_rev.npy'))
    X_test_adv_rev_mat = np.load(os.path.join(REV_DIR, 'X_test_adv_rev_mat.npy'))
    assert X_test_rev.shape[0] == test_size
    assert X_test_rev_mat.shape[0] == test_size
    assert X_test_adv_rev.shape[0] == test_size
    assert X_test_adv_rev_mat.shape[0] == test_size
    X_test_rev_all = np.concatenate((np.expand_dims(X_test_rev, axis=1), X_test_rev_mat), axis=1)  # (N, 10, 3, 32, 32)
    X_test_adv_rev_all = np.concatenate((np.expand_dims(X_test_adv_rev, axis=1), X_test_adv_rev_mat), axis=1)  # (N, 10, 3, 32, 32)
    del X_test_rev, X_test_rev_mat, X_test_adv_rev, X_test_adv_rev_mat

    if not os.path.exists(os.path.join(REV_DIR, 'y_test_adv_cross_rev_logits.npy')):
        print('generating cross predictions for {} using ensemble in {}'.format(REV_DIR, ENSEMBLE_DIR))
        ensemble_paths = get_ensemble_paths(ENSEMBLE_DIR)
        y_cross_rev_logits     = np.empty((test_size, len(ensemble_paths), len(ensemble_paths), len(classes)), dtype=np.float32)
        y_adv_cross_rev_logits = np.empty_like(y_cross_rev_logits)

        for j, ckpt_file in enumerate(ensemble_paths):  # for network j
            global_state = torch.load(ckpt_file, map_location=torch.device(device))
            net.load_state_dict(global_state['best_net'])
            for i in range(X_test_rev_all.shape[1]):  # for image created from network i
                print('predicting images created from network i={} on network j={} (net path={})'.format(i, j, ckpt_file))
                y_cross_rev_logits[:, i, j]     = classifier.predict(X_test_rev_all[:, i], batch_size=batch_size)
                y_adv_cross_rev_logits[:, i, j] = classifier.predict(X_test_adv_rev_all[:, i], batch_size=batch_size)

        # return to original net
        global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
        net.load_state_dict(global_state['best_net'])

        np.save(os.path.join(REV_DIR, 'y_test_cross_rev_logits.npy'), y_cross_rev_logits)
        np.save(os.path.join(REV_DIR, 'y_test_adv_cross_rev_logits.npy'), y_adv_cross_rev_logits)
    else:
        y_cross_rev_logits     = np.load(os.path.join(REV_DIR, 'y_test_cross_rev_logits.npy'))
        y_adv_cross_rev_logits = np.load(os.path.join(REV_DIR, 'y_test_adv_cross_rev_logits.npy'))

    assert y_cross_rev_logits.shape[0] == test_size
    assert y_adv_cross_rev_logits.shape[0] == test_size

    y_cross_rev_preds     = softmax((1/T) * y_cross_rev_logits, axis=3)      # (N, 10, 10, #cls)
    y_adv_cross_rev_preds = softmax((1/T) * y_adv_cross_rev_logits, axis=3)  # (N, 10, 10, #cls)

# organize features
if load_main and not load_ensemble:  # use only main
    normal_input     = np.expand_dims(y_cross_preds[:, 0], 1)                                       # (N, 1, #cls)
    adv_input        = np.expand_dims(y_adv_cross_preds[:, 0], 1)                                   # (N, 1, #cls)
    if load_rev:
        normal_rev_input = np.expand_dims(y_cross_rev_preds[:, 0, 0], 1)                            # (N, 1, #cls)
        adv_rev_input    = np.expand_dims(y_adv_cross_rev_preds[:, 0, 0], 1)                        # (N, 1, #cls)
elif not load_main and load_ensemble:  # use only ensemble
    normal_input     = y_cross_preds[:, 1:]                                                         # (N, 9, #cls)
    adv_input        = y_adv_cross_preds[:, 1:]                                                     # (N, 9, #cls)
    if load_rev:
        normal_rev_input = np.reshape(y_cross_rev_preds[:, 1:, 1:], (test_size, -1, len(classes)))  # (N, 9, 9, #cls) -> (N, 81, #cls)
        adv_rev_input = np.reshape(y_adv_cross_rev_preds[:, 1:, 1:],(test_size, -1, len(classes)))  # (N, 9, 9, #cls) -> (N, 81, #cls)
elif load_main and load_ensemble:
    normal_input     = y_cross_preds                                                                # (N, 10, #cls)
    adv_input        = y_adv_cross_preds                                                            # (N, 10, #cls)
    if load_rev:
        normal_rev_input = np.reshape(y_cross_rev_preds, (test_size, -1, len(classes)))             # (N, 10, 10, #cls) -> (N, 100, #cls)
        adv_rev_input = np.reshape(y_adv_cross_rev_preds,(test_size, -1, len(classes)))             # (N, 10, 10, #cls) -> (N, 100, #cls)
else:
    raise AssertionError('load_main or load_ensemble must be True')

# inputs stats
normal_mean     = np.mean(normal_input, axis=1)
normal_std      = np.std(normal_input, axis=1)
if load_rev:
    normal_rev_mean = np.mean(normal_rev_input, axis=1)
    normal_rev_std  = np.std(normal_rev_input, axis=1)

adv_mean     = np.mean(adv_input, axis=1)
adv_std      = np.std(adv_input, axis=1)
if load_rev:
    adv_rev_mean = np.mean(adv_rev_input, axis=1)
    adv_rev_std  = np.std(adv_rev_input, axis=1)

# grads
# not implemented yet.

normal_features = None
adv_features = None

# combining the features
assert load_orig or load_rev
if load_orig:
    normal_features = add_feature(normal_features, normal_input)
    adv_features    = add_feature(adv_features   , adv_input)
if load_rev:
    normal_features = add_feature(normal_features, normal_rev_input)
    adv_features    = add_feature(adv_features   , adv_rev_input)

if args.method == 'simple':
    preds = normal_features.argmax(axis=2)
    defense_preds = np.apply_along_axis(majority_vote, axis=1, arr=preds)

    preds_adv = adv_features.argmax(axis=2)
    defense_preds_adv = np.apply_along_axis(majority_vote, axis=1, arr=preds_adv)

elif 'svm' in args.method:
    # assert load_main and load_ensemble

    if args.method == 'inference_svm':
        print('Analyzing inference SVM robustness')
        pass

    elif args.method == 'stats_svm':
        print('Analyzing stats SVM robustness')
        normal_features = None
        adv_features = None

        normal_features = add_feature(normal_features, normal_mean)
        normal_features = add_feature(normal_features, normal_std)
        if load_rev:
            normal_features = add_feature(normal_features, normal_rev_mean)
            normal_features = add_feature(normal_features, normal_rev_std)

        adv_features = add_feature(adv_features, adv_mean)
        adv_features = add_feature(adv_features, adv_std)
        if load_rev:
            adv_features = add_feature(adv_features, adv_rev_mean)
            adv_features = add_feature(adv_features, adv_rev_std)

    elif args.method == 'inference_stats_svm':
        print('Analyzing inference+stats SVM robustness')
        normal_features = normal_features.reshape((test_size, -1))
        normal_features = add_feature(normal_features, normal_mean)
        normal_features = add_feature(normal_features, normal_std)
        if load_rev:
            normal_features = add_feature(normal_features, normal_rev_mean)
            normal_features = add_feature(normal_features, normal_rev_std)

        adv_features = adv_features.reshape((test_size, -1))
        adv_features = add_feature(adv_features, adv_mean)
        adv_features = add_feature(adv_features, adv_std)
        if load_rev:
            adv_features = add_feature(adv_features, adv_rev_mean)
            adv_features = add_feature(adv_features, adv_rev_std)

    # common code for all SVM methods:
    normal_features = normal_features.reshape((test_size, -1))
    adv_features    = adv_features.reshape((test_size, -1))

    if train_normal and train_adv:
        input_features = np.concatenate((normal_features[f1_inds_val], adv_features[f1_inds_val]))
        input_labels = np.concatenate((y_test[f1_inds_val], y_test[f1_inds_val]))
    elif train_normal:
        input_features = normal_features[f1_inds_val]
        input_labels = y_test[f1_inds_val]
    else:
        input_features = adv_features[f1_inds_val]
        input_labels = y_test[f1_inds_val]

    test_normal_features = normal_features.copy()
    test_adv_features = adv_features.copy()

    if args.pca_dims != -1:
        pca = PCA(n_components=args.pca_dims, random_state=rand_gen)
        pca.fit(input_features)
        input_features       = pca.transform(input_features)
        test_normal_features = pca.transform(test_normal_features)
        test_adv_features    = pca.transform(test_adv_features)

    if len(classes) == 100:
        max_iter = 1000
    else:
        max_iter = 10000
    clf = LinearSVC(penalty='l2', loss='hinge', verbose=1, random_state=rand_gen, max_iter=max_iter)
    clf.fit(input_features, input_labels)
    defense_preds = clf.predict(test_normal_features)
    defense_preds_adv = clf.predict(test_adv_features)


acc_all = np.mean(defense_preds[test_inds] == y_test[test_inds])
acc_f1 = np.mean(defense_preds[f1_inds_test] == y_test[f1_inds_test])
acc_f2 = np.mean(defense_preds[f2_inds_test] == y_test[f2_inds_test])
acc_f3 = np.mean(defense_preds[f3_inds_test] == y_test[f3_inds_test])

acc_all_adv = np.mean(defense_preds_adv[test_inds] == y_test[test_inds])
acc_f1_adv = np.mean(defense_preds_adv[f1_inds_test] == y_test[f1_inds_test])
acc_f2_adv = np.mean(defense_preds_adv[f2_inds_test] == y_test[f2_inds_test])
acc_f3_adv = np.mean(defense_preds_adv[f3_inds_test] == y_test[f3_inds_test])

print('Accuracy for method={}, train_on={}, net_pool={}, img_pool={}. T={}, PCA_DIMS={}: all samples: {:.2f}/{:.2f}%. '
      'f1 samples: {:.2f}/{:.2f}%, f2 samples: {:.2f}/{:.2f}%, f3 samples: {:.2f}/{:.2f}%'
      .format(args.method, args.train_on, args.net_pool, args.img_pool, T, args.pca_dims, acc_all * 100,
              acc_all_adv * 100, acc_f1 * 100, acc_f1_adv * 100, acc_f2 * 100, acc_f2_adv * 100, acc_f3 * 100, acc_f3_adv * 100))

# Distinguish scenarios:
# 1) same_correct  : y_new == y_old, correct label. BLUE.
# 2) same_incorrect: y_new == y_old, incorrect label. BLACK
# 3) flip_correct  : y_new != y_old, correct. GREEN
# 4) flip_incorrect: y_new != y_old, incorrect. RED
# same_correct   = np.logical_and(defense_preds == y_test_adv_preds, defense_preds == y_test)
# same_incorrect = np.logical_and(defense_preds == y_test_adv_preds, defense_preds != y_test)
# flip_correct   = np.logical_and(defense_preds != y_test_adv_preds, defense_preds == y_test)
# flip_incorrect = np.logical_and(defense_preds != y_test_adv_preds, defense_preds != y_test)
#
# # filter only indices inside f1_inds
# same_correct[f0_inds]   = False
# same_incorrect[f0_inds] = False
# flip_correct[f0_inds]   = False
# flip_incorrect[f0_inds] = False
#
# # Fitting PCA
# pca = PCA(n_components=2, random_state=rand_gen)
# pca_test_features_embeddings = pca.fit_transform(test_features)
# plt.close('all')
# plt.figure(figsize=(5,5))
# plt.scatter(pca_test_features_embeddings[same_correct, 0]  , pca_test_features_embeddings[same_correct, 1]  , s=1, c='blue' , label='same_correct')
# plt.scatter(pca_test_features_embeddings[same_incorrect, 0], pca_test_features_embeddings[same_incorrect, 1], s=1, c='black', label='same_incorrect')
# plt.scatter(pca_test_features_embeddings[flip_correct, 0]  , pca_test_features_embeddings[flip_correct, 1]  , s=1, c='green', label='flip_correct')
# plt.scatter(pca_test_features_embeddings[flip_incorrect, 0], pca_test_features_embeddings[flip_incorrect, 1], s=1, c='red'  , label='flip_incorrect')
# plt.legend()
# plt.savefig('/home/gilad/plots/pca.png')

# print best T
# T_vec = np.arange(1, 24)
# f1_acc = [86.85, 87.29, 87.76, 88.03, 88.48, 88.46, 88.61, 88.56, 88.65, 88.59, 88.48, 88.44, 88.37,
#           88.31, 88.31, 88.27, 88.29, 88.20, 88.14, 88.06, 88.06, 87.99, 87.95]
# plt.plot(T_vec, f1_acc)
# plt.xlabel('Temperature')
# plt.ylabel('f1 accuracy')
# plt.savefig('/home/gilad/plots/f1_acc_vs_T_cross_inference.png')

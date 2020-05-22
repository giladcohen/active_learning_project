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
from active_learning_project.utils import boolean_string, majority_vote
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor
from scipy.special import softmax

import matplotlib.pyplot as plt
from art.classifiers import PyTorchClassifier

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/svhn/resnet34/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack', default='cw', type=str, help='checkpoint dir')
parser.add_argument('--targeted', default=True, type=boolean_string, help='use targeted attack')
parser.add_argument('--attack_dir', default='', type=str, help='attack directory')
parser.add_argument('--rev_dir', default='fgsm_minimal', type=str, help='reverse dir')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--method', default='ensemble', type=str, help='method of defense: ensemble, smart_ensemble, svm, cross_inference_svm')
parser.add_argument('--pool', default='all', type=str, help='networks pool: main, ensemble, all')
parser.add_argument('--train_on', default='adv', type=str, help='normal, adv, all')
parser.add_argument('--test_on', default='adv', type=str, help='normal, adv')
parser.add_argument('--temperature', default=1, type=float, help='normal, adv')
parser.add_argument('--pca_dims', default=-1, type=int, help='if not -1, apply PCA to svm with dims')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)

CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
if args.attack_dir != '':
    ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
else:
    ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack)
    if args.targeted:
        ATTACK_DIR = ATTACK_DIR + '_targeted'
if args.rev_dir != '':
    REV_DIR = os.path.join(ATTACK_DIR, 'rev', args.rev_dir)
else:
    REV_DIR = os.path.join(ATTACK_DIR, 'rev', args.rev)
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
test_size  = len(testloader.dataset)
test_inds  = np.arange(test_size)

X_test           = get_normalized_tensor(testloader, batch_size)
y_test           = np.asarray(testloader.dataset.targets)
y_test_preds     = np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy'))

X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
y_test_adv_preds = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))
if args.targeted:
    y_test_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))

# Model
print('==> Building model..')
if train_args['net'] == 'resnet34':
    net = ResNet34(num_classes=len(classes))
elif train_args['net'] == 'resnet101':
    net = ResNet101(num_classes=len(classes))
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

# what are the samples we care about? net_succ (not attack_succ. it is irrelevant)
f0_inds = []  # net_fail
f1_inds = []  # net_succ
f2_inds = []  # net_succ AND attack_flip
f3_inds = []  # net_succ AND attack_flip AND attack_succ

for i in range(test_size):
    f1 = y_test_preds[i] == y_test[i]
    f2 = f1 and y_test_preds[i] != y_test_adv_preds[i]
    if args.targeted:
        f3 = f2 and y_test_adv_preds[i] == y_test_adv[i]
    else:
        f3 = f2
    if f1:
        f1_inds.append(i)
    else:
        f0_inds.append(i)
    if f2:
        f2_inds.append(i)
    if f3:
        f3_inds.append(i)

f0_inds = np.asarray(f0_inds)
f1_inds = np.asarray(f1_inds)
f2_inds = np.asarray(f2_inds)
f3_inds = np.asarray(f3_inds)

print("Number of test samples: {}. #net_succ: {}. #net_succ_attack_flip: {}. #net_succ_attack_succ: {}"
      .format(test_size, len(f1_inds), len(f2_inds), len(f3_inds)))

# Selecting train and test subsets - to make fair comparison with SVM and ensemble
tot = len(f1_inds)
n_train = int(0.5 * tot)
n_test = tot - n_train
f1_train = rand_gen.choice(f1_inds, n_train, replace=False)
f1_train.sort()
f1_test  = np.asarray([ind for ind in f1_inds if ind not in f1_train])

f2_train = np.asarray([ind for ind in f2_inds if ind in f1_train])
f2_test  = np.asarray([ind for ind in f2_inds if ind not in f1_train])

f3_train = np.asarray([ind for ind in f3_inds if ind in f1_train])
f3_test  = np.asarray([ind for ind in f3_inds if ind not in f1_train])

# load all calculated features:
load_main     = args.pool in ['main', 'all']
load_ensemble = args.pool in ['ensemble', 'all']
train_normal  = args.train_on in ['normal', 'all']
train_adv     = args.train_on in ['adv', 'all']

# main
y_main_logits         = np.load(os.path.join(ATTACK_DIR, 'y_test_logits.npy'))
y_adv_main_logits     = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_logits.npy'))
# ensemble
y_net_logits          = np.load(os.path.join(ENSEMBLE_DIR_DUMP, 'y_test_net_logits_mat.npy'))
y_adv_net_logits      = np.load(os.path.join(ENSEMBLE_DIR_DUMP, 'y_test_adv_net_logits_mat.npy'))
# main rev
y_main_rev_logits     = np.load(os.path.join(REV_DIR, 'y_test_rev_logits.npy'))
y_adv_main_rev_logits = np.load(os.path.join(REV_DIR, 'y_test_adv_rev_logits.npy'))
# ensemble rev
y_net_rev_logits      = np.load(os.path.join(REV_DIR, 'y_test_net_rev_logits_mat.npy'))
y_adv_net_rev_logits  = np.load(os.path.join(REV_DIR, 'y_test_adv_net_rev_logits_mat.npy'))

# calculating preds:
# main
y_main_preds          = softmax((1/T) * y_main_logits, axis=1)          # (N, #class)
y_adv_main_preds      = softmax((1/T) * y_adv_main_logits, axis=1)      # (N, #class)
# ensemble
y_net_preds           = softmax((1/T) * y_net_logits, axis=2)           # (N, 9, #class)
y_adv_net_preds       = softmax((1/T) * y_adv_net_logits, axis=2)       # (N, 9, #class)
# main rev
y_main_rev_preds      = softmax((1/T) * y_main_rev_logits, axis=1)      # (N, #class)
y_adv_main_rev_preds  = softmax((1/T) * y_adv_main_rev_logits, axis=1)  # (N, #class)
# ensemble_rev
y_net_rev_preds       = softmax((1/T) * y_net_rev_logits, axis=2)       # (N, 9, #class)
y_adv_net_rev_preds   = softmax((1/T) * y_adv_net_rev_logits, axis=2)   # (N, 9, #class)

if not os.path.exists(os.path.join(REV_DIR, 'y_test_adv_cross_rev_logits.npy')):
    X_test_rev         = np.load(os.path.join(REV_DIR, 'X_test_rev.npy'))
    X_test_rev_mat     = np.load(os.path.join(REV_DIR, 'X_test_rev_mat.npy'))
    X_test_adv_rev     = np.load(os.path.join(REV_DIR, 'X_test_adv_rev.npy'))
    X_test_adv_rev_mat = np.load(os.path.join(REV_DIR, 'X_test_adv_rev_mat.npy'))

    X_test_rev_all     = np.concatenate((np.expand_dims(X_test_rev, axis=1), X_test_rev_mat), axis=1)  # (N, 10, 3, 32, 32)
    X_test_adv_rev_all = np.concatenate((np.expand_dims(X_test_adv_rev, axis=1), X_test_adv_rev_mat), axis=1)  # (N, 10, 3, 32, 32)
    del X_test_rev, X_test_rev_mat, X_test_adv_rev, X_test_adv_rev_mat

    print('generating cross predictions for {} using ensemble in {}'.format(REV_DIR, ENSEMBLE_DIR))
    checkpoint_dir_list = next(os.walk(ENSEMBLE_DIR))[1]
    checkpoint_dir_list.sort()

    y_cross_rev_logits     = np.empty((test_size, len(checkpoint_dir_list), len(checkpoint_dir_list), len(classes)), dtype=np.float32)
    y_adv_cross_rev_logits = np.empty_like(y_cross_rev_logits)

    for j, dir in enumerate(checkpoint_dir_list):  # for network j
        ckpt_file = os.path.join(ENSEMBLE_DIR, dir, 'ckpt.pth')
        global_state = torch.load(ckpt_file, map_location=torch.device(device))
        net.load_state_dict(global_state['best_net'])
        for i in range(X_test_rev_all.shape[1]):  # for image created from network i
            y_cross_rev_logits[:, i, j]     = classifier.predict(X_test_rev_all[:, i], batch_size=batch_size)
            y_adv_cross_rev_logits[:, i, j] = classifier.predict(X_test_adv_rev_all[:, i], batch_size=batch_size)

    np.save(os.path.join(REV_DIR, 'y_test_cross_rev_logits.npy'), y_cross_rev_logits)
    np.save(os.path.join(REV_DIR, 'y_test_adv_cross_rev_logits.npy'), y_adv_cross_rev_logits)
else:
    y_cross_rev_logits     = np.load(os.path.join(REV_DIR, 'y_test_cross_rev_logits.npy'))
    y_adv_cross_rev_logits = np.load(os.path.join(REV_DIR, 'y_test_adv_cross_rev_logits.npy'))

y_cross_logits     = np.concatenate((np.expand_dims(y_main_logits, axis=1), y_net_logits), axis=1)  # (N, 10, #class)
y_adv_cross_logits = np.concatenate((np.expand_dims(y_adv_main_logits, axis=1), y_adv_net_logits), axis=1)  # (N, 10, #class)

y_cross_preds         = softmax((1/T) * y_cross_logits, axis=2)          # (N, 10, #cls)
y_adv_cross_preds     = softmax((1/T) * y_adv_cross_logits, axis=2)      # (N, 10, #cls)
y_cross_rev_preds     = softmax((1/T) * y_cross_rev_logits, axis=3)      # (N, 10, 10, #cls)
y_adv_cross_rev_preds = softmax((1/T) * y_adv_cross_rev_logits, axis=3)  # (N, 10, 10, #cls)

def add_feature(x, x1):
    """Adding feature x1 to x"""
    if x is None:
        x = x1
    else:
        x = np.concatenate((x, x1), axis=1)
    return x


if args.method == 'ensemble':
    print('Analyzing ensemble robustness on {} images'.format(args.test_on))
    # collect relevant features:
    preds = None
    if args.test_on == 'normal':
        if load_main:
            preds = add_feature(preds, np.expand_dims(y_main_preds, 1))
        if load_ensemble:
            preds = add_feature(preds, y_net_preds)
    else:
        if load_main:
            preds = add_feature(preds, np.expand_dims(y_adv_main_preds, 1))
        if load_ensemble:
            preds = add_feature(preds, y_adv_net_preds)

    preds = preds.argmax(axis=2)
    defense_preds = np.apply_along_axis(majority_vote, axis=1, arr=preds)

elif args.method == 'smart_ensemble':
    assert load_main and load_ensemble

    normal_features = None
    normal_features = add_feature(normal_features, y_cross_preds)
    normal_features = add_feature(normal_features, y_cross_rev_preds.reshape((test_size, -1, len(classes))))

    adv_features = None
    adv_features = add_feature(adv_features, y_adv_cross_preds)
    adv_features = add_feature(adv_features, y_adv_cross_rev_preds.reshape((test_size, -1, len(classes))))

    if args.test_on == 'normal':
        preds = normal_features
    else:
        preds = adv_features
    preds = preds.argmax(axis=2)
    defense_preds = np.apply_along_axis(majority_vote, axis=1, arr=preds)

elif 'svm' in args.method:
    if args.method == 'svm':
        print('Analyzing SVM robustness on {} images'.format(args.test_on))
        normal_features = None

        if load_main:
            normal_features = add_feature(normal_features, np.expand_dims(y_main_preds, 1))
            normal_features = add_feature(normal_features, np.expand_dims(y_main_rev_preds, 1))
        if load_ensemble:
            normal_features = add_feature(normal_features, y_net_preds)
            normal_features = add_feature(normal_features, y_net_rev_preds)
        normal_features = normal_features.reshape((test_size, -1))

        adv_features = None
        if load_main:
            adv_features = add_feature(adv_features, np.expand_dims(y_adv_main_preds, 1))
            adv_features = add_feature(adv_features, np.expand_dims(y_adv_main_rev_preds, 1))
        if load_ensemble:
            adv_features = add_feature(adv_features, y_adv_net_preds)
            adv_features = add_feature(adv_features, y_adv_net_rev_preds)
        adv_features = adv_features.reshape((test_size, -1))

    elif args.method == 'stats_svm':
        print('Analyzing stats SVM robustness on {} images'.format(args.test_on))
        assert load_main and load_ensemble

        normal_features = None
        normal_features = add_feature(normal_features, np.expand_dims(y_main_preds, 1))
        normal_features = add_feature(normal_features, np.expand_dims(y_main_rev_preds, 1))

        net_means     = np.mean(y_net_preds, axis=1)
        net_means_rev = np.mean(y_net_rev_preds, axis=1)
        net_std       = np.std(y_net_preds, axis=1)
        net_std_rev   = np.std(y_net_rev_preds, axis=1)

        normal_features = add_feature(normal_features, np.expand_dims(net_means, 1))
        normal_features = add_feature(normal_features, np.expand_dims(net_means_rev, 1))
        normal_features = add_feature(normal_features, np.expand_dims(net_std, 1))
        normal_features = add_feature(normal_features, np.expand_dims(net_std_rev, 1))
        normal_features = normal_features.reshape((test_size, -1))

        adv_features = None
        adv_features = add_feature(adv_features, np.expand_dims(y_adv_main_preds, 1))
        adv_features = add_feature(adv_features, np.expand_dims(y_adv_main_rev_preds, 1))

        net_means     = np.mean(y_adv_net_preds, axis=1)
        net_means_rev = np.mean(y_adv_net_rev_preds, axis=1)
        net_std       = np.std(y_adv_net_preds, axis=1)
        net_std_rev   = np.std(y_adv_net_rev_preds, axis=1)

        adv_features = add_feature(adv_features, np.expand_dims(net_means, 1))
        adv_features = add_feature(adv_features, np.expand_dims(net_means_rev, 1))
        adv_features = add_feature(adv_features, np.expand_dims(net_std, 1))
        adv_features = add_feature(adv_features, np.expand_dims(net_std_rev, 1))
        adv_features = adv_features.reshape((test_size, -1))

    elif args.method == 'cross_inference_svm':
        assert load_main and load_ensemble

        normal_features = None
        normal_features = add_feature(normal_features, y_cross_preds)
        normal_features = add_feature(normal_features, y_cross_rev_preds.reshape((test_size, -1, len(classes))))
        normal_features = normal_features.reshape((test_size, -1))

        adv_features = None
        adv_features = add_feature(adv_features, y_adv_cross_preds)
        adv_features = add_feature(adv_features, y_adv_cross_rev_preds.reshape((test_size, -1, len(classes))))
        adv_features = adv_features.reshape((test_size, -1))

    elif args.method == 'cross_inference_svm_v2':
        assert load_main and load_ensemble
        normal_features = None
        normal_features = add_feature(normal_features, np.expand_dims(y_cross_preds[:, 0], axis=1))  # just the reg main
        normal_features = add_feature(normal_features, np.expand_dims(y_cross_rev_preds[:, 0, 0], axis=1))  # just the rev main

        preds_main_on_rev = y_cross_preds[:, 1:]
        main_mean = np.mean(preds_main_on_rev, axis=1)     # (10k x #cls)
        main_std  = np.std(preds_main_on_rev, axis=1)      # (10k x #cls)
        preds_rev_on_rev = y_cross_rev_preds[:, 1:, 1:]
        rev_mean = np.mean(preds_rev_on_rev, axis=(1, 2))  # (10k x #cls)
        rev_std  = np.std(preds_rev_on_rev, axis=(1, 2))   # (10k x #cls)

        normal_features = add_feature(normal_features, np.expand_dims(main_mean, axis=1))
        normal_features = add_feature(normal_features, np.expand_dims(main_std, axis=1))
        normal_features = add_feature(normal_features, np.expand_dims(rev_mean, axis=1))
        normal_features = add_feature(normal_features, np.expand_dims(rev_std, axis=1))
        normal_features = normal_features.reshape((test_size, -1))

        adv_features = None
        adv_features = add_feature(adv_features, np.expand_dims(y_adv_cross_preds[:, 0], axis=1))  # just the reg main
        adv_features = add_feature(adv_features, np.expand_dims(y_adv_cross_rev_preds[:, 0, 0], axis=1))  # just the rev main

        preds_main_on_rev = y_adv_cross_preds[:, 1:]
        main_mean = np.mean(preds_main_on_rev, axis=1)     # (10k x #cls)
        main_std  = np.std(preds_main_on_rev, axis=1)      # (10k x #cls)
        preds_rev_on_rev = y_adv_cross_rev_preds[:, 1:, 1:]
        rev_mean = np.mean(preds_rev_on_rev, axis=(1, 2))  # (10k x #cls)
        rev_std  = np.std(preds_rev_on_rev, axis=(1, 2))   # (10k x #cls)

        adv_features = add_feature(adv_features, np.expand_dims(main_mean, axis=1))
        adv_features = add_feature(adv_features, np.expand_dims(main_std, axis=1))
        adv_features = add_feature(adv_features, np.expand_dims(rev_mean, axis=1))
        adv_features = add_feature(adv_features, np.expand_dims(rev_std, axis=1))
        adv_features = adv_features.reshape((test_size, -1))

    elif args.method == 'all_stats_svm':
        print('Analyzing all stats SVM robustness on {} images'.format(args.test_on))
        assert load_ensemble  # can be without main

        normal_features = None
        net_preds = None
        net_preds_rev = None
        if load_main:
            net_preds     = add_feature(net_preds    , np.expand_dims(y_main_preds, 1))
            net_preds_rev = add_feature(net_preds_rev, np.expand_dims(y_main_rev_preds, 1))
        if load_ensemble:
            net_preds     = add_feature(net_preds, y_net_preds)
            net_preds_rev = add_feature(net_preds_rev, y_net_rev_preds)

        net_means     = np.mean(net_preds, axis=1)
        net_means_rev = np.mean(net_preds_rev, axis=1)
        net_std       = np.std(net_preds, axis=1)
        net_std_rev   = np.std(net_preds_rev, axis=1)

        normal_features = add_feature(normal_features, np.expand_dims(net_means, 1))
        normal_features = add_feature(normal_features, np.expand_dims(net_means_rev, 1))
        normal_features = add_feature(normal_features, np.expand_dims(net_std, 1))
        normal_features = add_feature(normal_features, np.expand_dims(net_std_rev, 1))
        normal_features = normal_features.reshape((test_size, -1))

        adv_features = None
        net_preds = None
        net_preds_rev = None
        if load_main:
            net_preds     = add_feature(net_preds    , np.expand_dims(y_adv_main_preds, 1))
            net_preds_rev = add_feature(net_preds_rev, np.expand_dims(y_adv_main_rev_preds, 1))
        if load_ensemble:
            net_preds     = add_feature(net_preds, y_adv_net_preds)
            net_preds_rev = add_feature(net_preds_rev, y_adv_net_rev_preds)

        net_means     = np.mean(net_preds, axis=1)
        net_means_rev = np.mean(net_preds_rev, axis=1)
        net_std       = np.std(net_preds, axis=1)
        net_std_rev   = np.std(net_preds_rev, axis=1)

        adv_features = add_feature(adv_features, np.expand_dims(net_means, 1))
        adv_features = add_feature(adv_features, np.expand_dims(net_means_rev, 1))
        adv_features = add_feature(adv_features, np.expand_dims(net_std, 1))
        adv_features = add_feature(adv_features, np.expand_dims(net_std_rev, 1))
        adv_features = adv_features.reshape((test_size, -1))

    # common code for all SVM methods:
    if train_normal and train_adv:
        input_features = np.concatenate((normal_features[f1_train], adv_features[f1_train]))
        input_labels = np.concatenate((y_test[f1_train], y_test[f1_train]))
    elif train_normal:
        input_features = normal_features[f1_train]
        input_labels = y_test[f1_train]
    else:
        input_features = adv_features[f1_train]
        input_labels = y_test[f1_train]

    if args.test_on == 'normal':
        test_features = normal_features
    else:
        test_features = adv_features

    if args.pca_dims != -1:
        pca = PCA(n_components=args.pca_dims, random_state=rand_gen)
        pca.fit(input_features)
        input_features = pca.transform(input_features)
        test_features  = pca.transform(test_features)

    clf = LinearSVC(penalty='l2', loss='hinge', verbose=1, random_state=rand_gen, max_iter=10000)
    clf.fit(input_features, input_labels)
    defense_preds = clf.predict(test_features)

acc_all = np.mean(defense_preds == y_test)
acc_f1 = np.mean(defense_preds[f1_test] == y_test[f1_test])
acc_f2 = np.mean(defense_preds[f2_test] == y_test[f2_test])
acc_f3 = np.mean(defense_preds[f3_test] == y_test[f3_test])
print('Accuracy for method={}, train_on={}, test_on={}, pool={}, T={}, PCA_DIMS={}: all samples: {:.2f}%. f1 samples: {:.2f}%, f2 samples: {:.2f}%, f3 samples: {:.2f}%'
      .format(args.method, args.train_on, args.test_on, args.pool, T, args.pca_dims, acc_all * 100, acc_f1 * 100, acc_f2 * 100, acc_f3 * 100))

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

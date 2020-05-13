'''Train CIFAR10 with PyTorch.'''
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
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack', default='cw', type=str, help='checkpoint dir')
parser.add_argument('--targeted', default=True, type=boolean_string, help='use targeted attack')
parser.add_argument('--attack_dir', default='', type=str, help='attack directory')
parser.add_argument('--rev_dir', default='deepfool', type=str, help='reverse dir')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--method', default='ensemble', type=str, help='method of defense: ensemble, svm')
parser.add_argument('--train_on', default='adv', type=str, help='normal, adv, all')
parser.add_argument('--test_on', default='adv', type=str, help='normal, adv')

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
ENSEMBLE_DIR_DUMP = os.path.join(ATTACK_DIR, 'ensemble')

batch_size = args.batch_size

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
    if f2:
        f2_inds.append(i)
    if f3:
        f3_inds.append(i)

f1_inds = np.asarray(f1_inds)
f2_inds = np.asarray(f2_inds)
f3_inds = np.asarray(f3_inds)

print("Number of test samples: {}. #net_succ: {}. #net_succ_attack_flip: {}. # net_succ_attack_succ: {}"
      .format(test_size, len(f1_inds), len(f2_inds), len(f3_inds)))

# Selecting train and test subsets
rand_gen = np.random.RandomState(seed=12345)
tot = len(f1_inds)
n_train = int(0.5 * tot)
n_test = tot - n_train
f1_train = rand_gen.choice(f1_inds, n_train, replace=False)
f1_train.sort()
f1_test = np.asarray([ind for ind in f1_inds if ind not in f1_train])

# collect features:
T = 0.05
y_main_logits     = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_logits.npy'))
y_net_logits      = np.load(os.path.join(ENSEMBLE_DIR_DUMP, 'y_test_net_logits_mat.npy'))
y_main_rev_logits = np.load(os.path.join(REV_DIR, 'y_test_rev_logits.npy'))
y_net_rev_logits  = np.load(os.path.join(REV_DIR, 'y_test_rev_logits_mat.npy'))

# apply temperature softmax to everything:
y_main_preds      = softmax(T * y_main_logits, axis=1)      # (N, #class)
y_net_preds       = softmax(T * y_net_logits, axis=2)       # (N, 9, #class)
y_main_rev_preds  = softmax(T * y_main_rev_logits, axis=1)  # (N, #class)
y_net_rev_preds   = softmax(T * y_net_rev_logits, axis=2)   # (N, 9, #class)

# construct features
x1 = np.expand_dims(y_main_logits, 1)
x2 = np.expand_dims(y_main_rev_preds, 1)
input_features = np.concatenate((x1, y_net_preds, x2, y_net_rev_preds), axis=1)
input_features = input_features.reshape((test_size, -1))

train_inputs = input_features[f1_train]
train_labels = y_test[f1_train]
test_inputs  = input_features[f1_test]
test_labels  = y_test[f1_test]


clf = LinearSVC(penalty='l2', loss='hinge', verbose=1, random_state=rand_gen, max_iter=100000)
clf.fit(train_inputs, train_labels)
y_preds = clf.predict(test_inputs)
accuracy = np.mean(y_preds == test_labels)
print('accuracy = {}'.format(accuracy))


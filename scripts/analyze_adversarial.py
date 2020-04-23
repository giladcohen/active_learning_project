'''Train CIFAR10 with PyTorch.'''
import torch
import numpy as np
import json
import os
import argparse

from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_loader_with_specific_inds
from active_learning_project.utils import convert_tensor_to_image
from active_learning_project.utils import boolean_string

import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack', default='fgsm', type=str, help='checkpoint dir')
parser.add_argument('--targeted', default=True, type=boolean_string, help='use targeted attack')
parser.add_argument('--rev_dir', default='smart_ensemble_fgsm', type=str, help='reverse dir')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
DATA_ROOT = '/data/dataset/cifar10'
ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack)
if args.targeted:
    ATTACK_DIR = ATTACK_DIR + '_targeted'
REV_DIR = os.path.join(ATTACK_DIR, 'rev', args.rev_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
train_inds = global_state['train_inds']
val_inds = global_state['val_inds']
test_inds = np.arange(10000).tolist()
batch_size = 100

# load data
print('==> Preparing data..')
trainloader = get_loader_with_specific_inds(
    data_dir=DATA_ROOT,
    batch_size=batch_size,
    is_training=False,
    indices=train_inds,
    num_workers=1,
    pin_memory=True
)
valloader = get_loader_with_specific_inds(
    data_dir=DATA_ROOT,
    batch_size=batch_size,
    is_training=False,
    indices=val_inds,
    num_workers=1,
    pin_memory=True
)
testloader = get_test_loader(
    data_dir=DATA_ROOT,
    batch_size=batch_size,
    num_workers=1,
    pin_memory=True
)

classes = trainloader.dataset.classes
train_size = len(trainloader.dataset)
val_size   = len(valloader.dataset)
test_size  = len(testloader.dataset)

X_val            = valloader.dataset.data
y_val            = valloader.dataset.targets
y_val_preds      = np.load(os.path.join(ATTACK_DIR, 'y_val_preds.npy'))
X_val_adv        = np.load(os.path.join(ATTACK_DIR, 'X_val_adv.npy'))
y_val_adv_preds  = np.load(os.path.join(ATTACK_DIR, 'y_val_adv_preds.npy'))

X_test           = testloader.dataset.data
y_test           = testloader.dataset.targets
y_test_preds     = np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy'))
X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
y_test_adv_preds = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))

if 'ensemble' not in REV_DIR:
    X_test_rev       = np.load(os.path.join(REV_DIR   , 'X_test_rev.npy'))
y_test_rev_preds = np.load(os.path.join(REV_DIR, 'y_test_rev_preds.npy'))

if args.targeted:
    y_val_adv  = np.load(os.path.join(ATTACK_DIR, 'y_val_adv.npy'))
    y_test_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))

# get stats:
info = {}
info['val'] = {}
for i, set_ind in enumerate(val_inds):
    info['val'][i] = {}
    net_succ = y_val_preds[i] == y_val[i]
    attack_flipped = y_val_preds[i] != y_val_adv_preds[i]
    if args.targeted:
        attack_succ = attack_flipped and y_val_adv_preds[i] == y_val_adv[i]
    else:
        attack_succ = attack_flipped
    info['val'][i]['global_index'] = set_ind
    info['val'][i]['net_succ'] = net_succ
    info['val'][i]['attack_flipped'] = attack_flipped
    info['val'][i]['attack_succ'] = attack_succ
info['test'] = {}
for i, set_ind in enumerate(test_inds):
    info['test'][i] = {}
    net_succ = y_test_preds[i] == y_test[i]
    attack_flipped = y_test_preds[i] != y_test_adv_preds[i]
    if args.targeted:
        attack_succ = attack_flipped and y_test_adv_preds[i] == y_test_adv[i]
    else:
        attack_succ = attack_flipped
    rev_known = y_test_rev_preds[i] != -1
    rev_flip = rev_known and y_test_rev_preds[i] != y_test_adv_preds[i]
    rev_succ = rev_flip and y_test_rev_preds[i] == y_test[i]
    info['test'][i]['global_index'] = set_ind
    info['test'][i]['net_succ'] = net_succ
    info['test'][i]['attack_flipped'] = attack_flipped
    info['test'][i]['attack_succ'] = attack_succ
    info['test'][i]['rev_known'] = rev_known
    info['test'][i]['rev_flip'] = rev_flip
    info['test'][i]['rev_succ'] = rev_succ

f_inds = [ind for ind in info['test'] if info['test'][ind]['net_succ'] and info['test'][ind]['attack_succ']]

# assert that if net_succ and attack_succ then rev_known is always True
filtered_rev_known = [info['test'][ind]['rev_known'] for ind in f_inds]
assert np.array(filtered_rev_known).all(), 'We expect that if net_succ and attack_succ then the rev label will be known'

val_acc = np.sum(y_val_preds == y_val) / val_size
test_acc = np.sum(y_test_preds == y_test) / test_size
print('Accuracy on benign val examples: {}%'.format(val_acc * 100))
print('Accuracy on benign test examples: {}%'.format(test_acc * 100))

val_adv_accuracy = np.sum(y_val_adv_preds == y_val) / val_size
test_adv_accuracy = np.sum(y_test_adv_preds == y_test) / test_size
print('Accuracy on adversarial val examples: {}%'.format(val_adv_accuracy * 100))
print('Accuracy on adversarial test examples: {}%'.format(test_adv_accuracy * 100))

val_net_succ_indices = [ind for ind in info['val'] if info['val'][ind]['net_succ']]
val_net_succ_attack_succ_indices = [ind for ind in info['val'] if info['val'][ind]['net_succ'] and info['val'][ind]['attack_succ']]
test_net_succ_indices = [ind for ind in info['test'] if info['test'][ind]['net_succ']]
test_net_succ_attack_succ_indices = [ind for ind in info['test'] if info['test'][ind]['net_succ'] and info['test'][ind]['attack_succ']]
val_attack_rate = len(val_net_succ_attack_succ_indices) / len(val_net_succ_indices)
test_attack_rate = len(test_net_succ_attack_succ_indices) / len(test_net_succ_indices)
print('adversarial validation attack rate: {}\nadversarial test attack rate: {}'.format(val_attack_rate, test_attack_rate))

wrong_flip_indices = [ind for ind in info['test'] if info['test'][ind]['net_succ'] and info['test'][ind]['attack_flipped'] and not info['test'][ind]['attack_succ']]
right_flip_indices = [ind for ind in info['test'] if info['test'][ind]['net_succ'] and info['test'][ind]['attack_succ']]
print('out of {} prediction flips, only {} are flips towards the adversarial label'.format(len(wrong_flip_indices) + len(right_flip_indices), len(right_flip_indices)))

rev_flip_indices = [ind for ind in info['test'] if info['test'][ind]['net_succ'] and info['test'][ind]['attack_succ'] \
                    and info['test'][ind]['rev_flip']]
rev_succ_indices = [ind for ind in info['test'] if info['test'][ind]['net_succ'] and info['test'][ind]['attack_succ'] \
                    and info['test'][ind]['rev_succ']]
print('out of {} successful attacks, we reverted {} samples. Successful number of reverted: {}'.format(len(right_flip_indices), len(rev_flip_indices), len(rev_succ_indices)))

# DEBUG
# convert adv to BRGB:
# X_val_adv  = convert_tensor_to_image(X_val_adv)
# X_test_adv = convert_tensor_to_image(X_test_adv)
# if 'ensemble' not in REV_DIR:
#     X_test_rev = convert_tensor_to_image(X_test_rev)
#
# i = 6
# plt.figure(1)
# plt.imshow(X_test[i])
# plt.show()
# plt.figure(2)
# plt.imshow(X_test_adv[i])
# plt.show()
# if 'ensemble' not in REV_DIR:
#     plt.figure(3)
#     plt.imshow(X_test_rev[i])
#     plt.show()
#
# if args.targeted:
#     print('class is {}, model predicted {}, we wanted to attack to {}, and after adv noise: {}. After reverse: {}'
#         .format(classes[y_test[i]], classes[y_test_preds[i]], classes[y_test_adv[i]], classes[y_test_adv_preds[i]], classes[y_test_rev_preds[i]]))
# else:
#     print('class is {}, model predicted {}, and after adv noise: {}. After reverse: {}'
#         .format(classes[y_test[i]], classes[y_test_preds[i]], classes[y_test_adv_preds[i]], classes[y_test_rev_preds[i]]))


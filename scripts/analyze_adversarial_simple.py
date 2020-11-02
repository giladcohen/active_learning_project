'''Train CIFAR10 with PyTorch.'''
import torch
import numpy as np
import json
import os
import argparse

from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_loader_with_specific_inds
from active_learning_project.utils import convert_tensor_to_image
from active_learning_project.utils import boolean_string

import matplotlib
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='PyTorch adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/Users/giladcohen/data/gilad/logs/adv_robustness/cifar100/resnet34/regular_softplus/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='deepfool', type=str, help='attack directory')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)

ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
batch_size = args.batch_size

# load data
print('==> Preparing data..')
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
train_inds = np.asarray(global_state['train_inds'])
val_inds = np.asarray(global_state['val_inds'])
trainloader = get_loader_with_specific_inds(
    dataset=train_args['dataset'],
    batch_size=batch_size,
    is_training=False,
    indices=train_inds,
    num_workers=1,
    pin_memory=True
)
valloader = get_loader_with_specific_inds(
    dataset=train_args['dataset'],
    batch_size=batch_size,
    is_training=False,
    indices=val_inds,
    num_workers=1,
    pin_memory=True
)
testloader = get_test_loader(
    dataset=train_args['dataset'],
    batch_size=batch_size,
    num_workers=1,
    pin_memory=True
)

classes = testloader.dataset.classes
train_size = len(trainloader.dataset)
val_size   = len(valloader.dataset)
test_size  = len(testloader.dataset)
test_inds  = np.arange(test_size)

X_val            = valloader.dataset.data
y_val            = np.asarray(valloader.dataset.targets)
y_val_preds      = np.load(os.path.join(ATTACK_DIR, 'y_val_preds.npy'))
X_val_adv        = np.load(os.path.join(ATTACK_DIR, 'X_val_adv.npy'))
y_val_adv_preds  = np.load(os.path.join(ATTACK_DIR, 'y_val_adv_preds.npy'))

X_test           = testloader.dataset.data
y_test           = np.asarray(testloader.dataset.targets)
y_test_preds     = np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy'))
X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
y_test_adv_preds = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))

if targeted:
    y_val_adv  = np.load(os.path.join(ATTACK_DIR, 'y_val_adv.npy'))
    y_test_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))

# get stats:
info = {}
info['val'] = {}
for i, set_ind in enumerate(val_inds):
    info['val'][i] = {}
    net_succ = y_val_preds[i] == y_val[i]
    attack_flipped = y_val_preds[i] != y_val_adv_preds[i]
    if targeted:
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
    if targeted:
        attack_succ = attack_flipped and y_test_adv_preds[i] == y_test_adv[i]
    else:
        attack_succ = attack_flipped
    info['test'][i]['global_index'] = set_ind
    info['test'][i]['net_succ'] = net_succ
    info['test'][i]['attack_flipped'] = attack_flipped
    info['test'][i]['attack_succ'] = attack_succ

f_inds = [ind for ind in info['test'] if info['test'][ind]['net_succ'] and info['test'][ind]['attack_succ']]

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

# DEBUG
# convert adv to BRGB:
X_val_adv  = convert_tensor_to_image(X_val_adv)
X_test_adv = convert_tensor_to_image(X_test_adv)

N = 5
ROWS = 2
inds = np.random.choice(f_inds, N)
fig = plt.figure(figsize=(N, ROWS))
for i in range(N):
    fig.add_subplot(ROWS, N, i+1)
    plt.imshow(X_test[inds[i]])
    plt.axis('off')
    fig.add_subplot(ROWS, N, i + N + 1)
    plt.imshow(X_test_adv[inds[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()

i = inds[0]
strr = 'class is {}({}), model predicted {}({}), '.format(classes[y_test[i]], y_test[i], classes[y_test_preds[i]], y_test_preds[i])
if targeted:
    strr += 'we wanted to attack to {}({}), '.format(classes[y_test_adv[i]], y_test_adv[i])
strr += 'and after adv noise: {}({}).\n'.format(classes[y_test_adv_preds[i]], y_test_adv_preds[i])
print(strr)

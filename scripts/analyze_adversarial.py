'''Train CIFAR10 with PyTorch.'''
import torch
import numpy as np
import json
import os
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_loader_with_specific_inds
from active_learning_project.utils import convert_tensor_to_image
import matplotlib.pyplot as plt
import pickle

CHECKPOINT_DIR = '/data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00'
with open(os.path.join(CHECKPOINT_DIR, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
ATTACK = 'fgsm_targeted'
ATTACK_DIR = os.path.join(CHECKPOINT_DIR, ATTACK)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'ckpt.pth')
DATA_ROOT = '/data/dataset/cifar10'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(os.path.join(CHECKPOINT_DIR, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
train_inds = global_state['train_inds']
val_inds = global_state['val_inds']
test_inds = np.arange(10000).tolist()
batch_size = 100

# load data
# Data
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

X_val          = valloader.dataset.data
y_val          = valloader.dataset.targets
val_preds      = np.load(os.path.join(ATTACK_DIR, 'val_preds.npy'))
X_val_adv      = np.load(os.path.join(ATTACK_DIR, 'X_val_adv.npy'))
val_adv_preds  = np.load(os.path.join(ATTACK_DIR, 'val_adv_preds.npy'))

X_test         = testloader.dataset.data
y_test         = testloader.dataset.targets
test_preds     = np.load(os.path.join(ATTACK_DIR, 'test_preds.npy'))
X_test_adv     = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
test_adv_preds = np.load(os.path.join(ATTACK_DIR, 'test_adv_preds.npy'))

if 'targeted' in ATTACK:
    y_val_adv = np.load(os.path.join(ATTACK_DIR, 'y_val_adv.npy'))
    y_test_adv     = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))

# convert adv to BRGB:
X_val_adv  = convert_tensor_to_image(X_val_adv)
X_test_adv = convert_tensor_to_image(X_test_adv)

# get stats:
info_file = os.path.join(ATTACK_DIR, 'info.pkl')
print('loading info as pickle from {}'.format(info_file))
with open(info_file, 'rb') as handle:
    info = pickle.load(handle)

val_acc = np.sum(val_preds == y_val) / val_size
test_acc = np.sum(test_preds == y_test) / test_size
print('Accuracy on benign val examples: {}%'.format(val_acc * 100))
print('Accuracy on benign test examples: {}%'.format(test_acc * 100))

val_adv_accuracy = np.sum(val_adv_preds == y_val) / val_size
test_adv_accuracy = np.sum(test_adv_preds == y_test) / test_size
print('Accuracy on adversarial val examples: {}%'.format(val_adv_accuracy * 100))
print('Accuracy on adversarial test examples: {}%'.format(test_adv_accuracy * 100))

# calculate number of net_succ
val_net_succ_indices = [ind for ind in info['val'] if info['val'][ind]['net_succ']]
val_net_succ_attack_succ_indices = [ind for ind in info['val'] if info['val'][ind]['net_succ'] and info['val'][ind]['attack_succ']]
test_net_succ_indices = [ind for ind in info['test'] if info['test'][ind]['net_succ']]
test_net_succ_attack_succ_indices = [ind for ind in info['test'] if info['test'][ind]['net_succ'] and info['test'][ind]['attack_succ']]
val_attack_rate = len(val_net_succ_attack_succ_indices) / len(val_net_succ_indices)
test_attack_rate = len(test_net_succ_attack_succ_indices) / len(test_net_succ_indices)
print('adversarial validation attack rate: {}\nadversarial test attack rate: {}'.format(val_attack_rate, test_attack_rate))

i = 6
plt.figure(1)
plt.imshow(X_val[i])
plt.show()
plt.figure(2)
plt.imshow(X_val_adv[i])
plt.show()
print('class is {}, model predicted {}, and after adv noise: {}'
    .format(classes[y_val[i]], classes[val_preds[i]], classes[val_adv_preds[i]]))

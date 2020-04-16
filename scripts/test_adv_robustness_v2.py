'''Train CIFAR10 with PyTorch.'''
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary

import numpy as np
import json
import os

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from active_learning_project.utils import convert_tensor_to_image
import matplotlib.pyplot as plt
import pickle

from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier


CHECKPOINT_DIR = '/data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00'
with open(os.path.join(CHECKPOINT_DIR, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
attack = 'fgsm'
targeted = True
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'ckpt.pth')
DATA_ROOT = '/data/dataset/cifar10'
ATTACK_DIR = os.path.join(CHECKPOINT_DIR, attack)
if targeted:
    ATTACK_DIR = ATTACK_DIR + '_targeted'
REV_DIR = os.path.join(ATTACK_DIR, 'rev')
os.makedirs(REV_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(os.path.join(CHECKPOINT_DIR, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
train_inds = global_state['train_inds']
val_inds = global_state['val_inds']
test_inds = np.arange(10000).tolist()
batch_size = 100

# Data
print('==> Preparing data..')
testloader = get_test_loader(
    data_dir=DATA_ROOT,
    batch_size=batch_size,
    num_workers=1,
    pin_memory=True
)

classes = testloader.dataset.classes
test_size  = len(testloader.dataset)

X_test           = get_normalized_tensor(testloader, batch_size)
y_test           = testloader.dataset.targets

X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
if targeted:
    y_test_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))

# Model
print('==> Building model..')
if train_args['net'] == 'resnet34':
    net = ResNet34()
elif train_args['net'] == 'resnet101':
    net = ResNet101()
else:
    raise AssertionError("network {} is unknown".format(train_args['net']))
net = net.to(device)

summary(net, (3, 32, 32))
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
                               optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10)

y_test_preds = classifier.predict(X_test, batch_size=batch_size).argmax(axis=1)
assert (y_test_preds == np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy'))).all()

y_test_adv_preds = classifier.predict(X_test_adv, batch_size=batch_size).argmax(axis=1)
assert (y_test_adv_preds == np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))).all()

# reverse attack:
attack = FastGradientMethod(
    classifier=classifier,
    norm=np.inf,
    eps=0.01,
    eps_step=0.003,
    targeted=False,
    num_random_init=0,
    batch_size=batch_size
)

X_test_rev = attack.generate(x=X_test_adv)
y_test_rev_preds = classifier.predict(X_test_rev, batch_size=batch_size).argmax(axis=1)
np.save(os.path.join(REV_DIR, 'X_test_rev.npy'), X_test_rev)
np.save(os.path.join(REV_DIR, 'y_test_rev_preds.npy'), y_test_rev_preds)



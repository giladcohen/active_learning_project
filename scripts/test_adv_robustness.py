'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import time

import sys
sys.path.insert(0, ".")

from active_learning_project.models.resnet_v2 import ResNet18
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds
from active_learning_project.datasets.selection_methods import select_random, update_inds, SelectionMethodFactory
from active_learning_project.utils import remove_substr_from_keys
from torchsummary import summary
from active_learning_project.utils import boolean_string, pytorch_evaluate
from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/disk4/dynamic_wd/dynamic_wd_0.00039_mom_0.9', type=str, help='checkpoint dir')
parser.add_argument('--attack', default='fgsm', type=str, help='checkpoint dir')
parser.add_argument('--force_bn', default=True, type=boolean_string, help='hack to force batch norm')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/data/dataset/cifar10'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
train_inds = global_state['train_inds']
val_inds = global_state['val_inds']

# Data
print('==> Preparing data..')
trainloader = get_loader_with_specific_inds(
    data_dir=DATA_ROOT,
    batch_size=100,
    is_training=True,
    indices=train_inds,
    num_workers=1,
    pin_memory=True
)
valloader = get_loader_with_specific_inds(
    data_dir=DATA_ROOT,
    batch_size=100,
    is_training=False,
    indices=val_inds,
    num_workers=1,
    pin_memory=True
)
testloader = get_test_loader(
    data_dir=DATA_ROOT,
    batch_size=100,
    num_workers=1,
    pin_memory=True
)

classes = trainloader.dataset.dataset.classes
train_size = len(trainloader.dataset)
val_size   = len(valloader.dataset)
test_size  = len(testloader.dataset)

# Model
print('==> Building model..')
use_bn = (train_args.get('use_bn') == True or args.force_bn)
net = ResNet18(num_classes=len(classes), use_bn=use_bn)
net = net.to(device)
# summary(net, (3, 32, 32))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net.load_state_dict(global_state['best_net'])

criterion = nn.CrossEntropyLoss()

if __name__ == "__main__":
    (test_logits,) = pytorch_evaluate(net, testloader, ['logits'])
    test_preds = np.argmax(test_logits, axis=1)
    test_acc = np.sum(test_preds == testloader.dataset.targets) / test_size
    print('Accuracy on benign test examples: {}%'.format(test_acc * 100))

    # attack
    classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=criterion,
                                   optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10)
    attack = FastGradientMethod(classifier=classifier, eps=0.2)





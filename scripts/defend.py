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
sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from active_learning_project.utils import convert_tensor_to_image
from active_learning_project.utils import boolean_string

import matplotlib.pyplot as plt

from art.attacks import FastGradientMethod, ProjectedGradientDescent, DeepFool
from art.classifiers import PyTorchClassifier

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack', default='fgsm', type=str, help='checkpoint dir')
parser.add_argument('--targeted', default=True, type=boolean_string, help='use trageted attack')
parser.add_argument('--rev', type=str, help='fgsm, pgd, deepfool, ensemble')
parser.add_argument('--rev_dir', default='', type=str, help='reverse dir')
parser.add_argument('--ensemble_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34', type=str, help='ensemble dir of many networks')

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
if args.rev_dir != '':
    REV_DIR = os.path.join(ATTACK_DIR, 'rev', args.rev_dir)
else:
    REV_DIR = os.path.join(ATTACK_DIR, 'rev', args.rev)
os.makedirs(REV_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
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
if args.targeted:
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
if args.rev == 'fgsm':
    attack = FastGradientMethod(
        classifier=classifier,
        norm=np.inf,
        eps=0.01,
        eps_step=0.003,
        targeted=False,
        num_random_init=0,
        batch_size=batch_size
    )
elif args.rev == 'pgd':
    attack = ProjectedGradientDescent(
        classifier=classifier,
        norm=np.inf,
        eps=0.01,
        eps_step=0.003,
        targeted=False,
        batch_size=batch_size
    )
elif args.rev == 'deepfool':
    attack = DeepFool(
        classifier=classifier,
        epsilon=1e-6,
        nb_grads=len(classes),
        batch_size=batch_size
    )
elif args.rev == 'ensemble':
    print('Running ensemble defense. Loading all models')
    checkpoint_dir_list = next(os.walk(args.ensemble_dir))[1]
    checkpoint_dir_list.sort()
    checkpoint_dir_list = checkpoint_dir_list[1:]  # ignoring the first (original) network
    y_test_pred_mat = -1 * np.ones((test_size, len(checkpoint_dir_list)), dtype=np.int32)
    for i, dir in tqdm(enumerate(checkpoint_dir_list)):
        ckpt_file = os.path.join(args.ensemble_dir, dir, 'ckpt.pth')
        global_state = torch.load(ckpt_file, map_location=torch.device(device))
        net.load_state_dict(global_state['best_net'])
        print('fetching predictions using ckpt file: {}'.format(ckpt_file))
        y_test_pred_mat[:, i] = classifier.predict(X_test_adv, batch_size=batch_size).argmax(axis=1)
    assert not (y_test_pred_mat == -1).any()

    def majority_vote(x):
        return np.bincount(x).argmax()

    y_test_rev_preds = np.apply_along_axis(majority_vote, axis=1, arr=y_test_pred_mat)
else:
    raise AssertionError('Unknown rev {}'.format(args.rev))

if args.rev != 'ensemble':
    X_test_rev = attack.generate(x=X_test_adv)
    y_test_rev_preds = classifier.predict(X_test_rev, batch_size=batch_size).argmax(axis=1)
    np.save(os.path.join(REV_DIR, 'X_test_rev.npy'), X_test_rev)
np.save(os.path.join(REV_DIR, 'y_test_rev_preds.npy'), y_test_rev_preds)



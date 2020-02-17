'''Test CIFAR10 robustness with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import json
import os
import argparse
import time

import sys
sys.path.insert(0, ".")

from active_learning_project.models.resnet_v2 import ResNet18
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds
from torchsummary import summary
from active_learning_project.utils import boolean_string, pytorch_evaluate
from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier
from cleverhans.utils import random_targets, to_categorical
from torchvision import transforms


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/disk4/dynamic_wd/simple_wd_0.00039_mom_0.9_160220', type=str, help='checkpoint dir')
parser.add_argument('--attack', default='fgsm', type=str, help='checkpoint dir')
parser.add_argument('--targeted', default=True, type=boolean_string, help='use trageted attack')
parser.add_argument('--force_bn', default=True, type=boolean_string, help='hack to force batch norm')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/data/dataset/cifar10'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack)
if args.targeted:
    ATTACK_DIR = ATTACK_DIR + '_targeted'
os.makedirs(ATTACK_DIR, exist_ok=True)

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
train_inds = global_state['train_inds']
val_inds = global_state['val_inds']
batch_size = 100

# Data
print('==> Preparing data..')
trainloader = get_loader_with_specific_inds(
    data_dir=DATA_ROOT,
    batch_size=batch_size,
    is_training=True,
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

classes = trainloader.dataset.dataset.classes
train_size = len(trainloader.dataset)
val_size   = len(valloader.dataset)
test_size  = len(testloader.dataset)

# Model
print('==> Building model..')
use_bn = (train_args.get('use_bn') == True or args.force_bn)
net = ResNet18(num_classes=len(classes), use_bn=use_bn, return_logits_only=True)
net = net.to(device)

# summary(net, (3, 32, 32))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
# net.load_state_dict(global_state['best_net'])
net.load_state_dict(global_state['best_net'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=train_args['lr'],
    momentum=train_args['mom'],
    weight_decay=0.0,
    nesterov=train_args['mom'] > 0)

# if __name__ == "__main__":

X_test     = -1.0 * np.ones(shape=(test_size, 3, 32, 32), dtype=np.float32)
y_test = testloader.dataset.targets
# test_preds = -1 * np.ones(test_size, dtype=np.int)

# net.eval()
net.eval()
classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=criterion,
                               optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10)

for batch_idx, (inputs, targets) in enumerate(testloader):
    b = batch_idx * batch_size
    e = b + targets.shape[0]
    X_test[b:e] = inputs.cpu().numpy()

test_preds = classifier.predict(X_test, batch_size=batch_size)
test_preds = test_preds.argmax(axis=1)
test_acc = np.sum(test_preds == y_test) / test_size
print('Accuracy on benign test examples: {}%'.format(test_acc * 100))

# (X_test, test_logits) = pytorch_evaluate(net, testloader, ['images', 'logits'])
# test_preds2 = np.argmax(test_logits, axis=1)
# test_acc = np.sum(test_preds2 == test_labels) / test_size
# print('Accuracy on benign test examples: {}%'.format(test_acc * 100))

# attack
# creating targeted labels
if args.targeted:
    tgt_file = os.path.join(ATTACK_DIR, 'y_test_targets.npy')
    if not os.path.isfile(tgt_file):
        y_test_targets = random_targets(np.asarray(y_test), len(classes))  # .argmax(axis=1)
        np.save(tgt_file, y_test_targets.argmax(axis=1))
    else:
        y_test_targets = np.load(tgt_file)
        y_test_targets = to_categorical(y_test_targets, nb_classes=len(classes))
else:
    y_test_targets = None

if args.attack == 'fgsm':
    attack = FastGradientMethod(
        classifier=classifier,
        norm=np.inf,
        eps=0.3,
        eps_step=0.1,
        targeted=args.targeted,
        num_random_init=0,
        batch_size=batch_size
    )
else:
    err_str = print('Attack {} is not supported'.format(args.attack))
    print(err_str)
    raise AssertionError(err_str)

X_test_adv = attack.generate(x=X_test, y=y_test_targets)
test_adv_logits = classifier.predict(X_test_adv, batch_size=batch_size)
test_adv_preds = np.argmax(test_adv_logits, axis=1)
adv_accuracy = np.sum(test_adv_preds == y_test) / test_size
print('Accuracy on adversarial test examples: {}%'.format(adv_accuracy * 100))

# view X_adv
# np.save(os.path.join(ATTACK_DIR, 'X_test_adv_eps_0.6_step_0.3.npy'), X_test_adv)
# unorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

inv_normalize = transforms.Normalize(
    mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
    std=[1/0.2023, 1/0.1994, 1/0.2010]
)
X_test_normalized     = np.zeros((test_size, 32, 32, 3))
X_test_adv_normalized = np.zeros((test_size, 32, 32, 3))
for i in range(test_size):
    tmp = inv_normalize(torch.tensor(X_test[i])).cpu().numpy()
    X_test_normalized[i] = np.swapaxes(tmp, 2, 0)
    tmp = inv_normalize(torch.tensor(X_test_adv[i])).cpu().numpy()
    X_test_adv_normalized[i] = np.swapaxes(tmp, 2, 0)

X_test_normalized_img     = (np.round(X_test_normalized * 255.0)).astype(np.int)
X_test_adv_normalized_img = (np.round(X_test_adv_normalized * 255.0)).astype(np.int)
# np.save(os.path.join(ATTACK_DIR, 'X_test_img.npy'), X_test_normalized_img)
np.save(os.path.join(ATTACK_DIR, 'X_test_adv_img.npy_eps_0.3_step_0.1'), X_test_adv_normalized_img)




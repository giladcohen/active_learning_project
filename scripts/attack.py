'''Test CIFAR10 robustness with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary

import numpy as np
import json
import os
import argparse
import time
import pickle

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")


from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from active_learning_project.utils import boolean_string, pytorch_evaluate
from art.attacks import FastGradientMethod, ProjectedGradientDescent, DeepFool, SaliencyMapMethod, CarliniL2Method, \
    ElasticNet
from art.classifiers import PyTorchClassifier
from cleverhans.utils import random_targets, to_categorical

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack', default='fgsm', type=str, help='attack: fgsm, jsma, cw, deepfool, ead, pgd')
parser.add_argument('--targeted', default=True, type=boolean_string, help='use trageted attack')
parser.add_argument('--attack_dir', default='', type=str, help='attack directory')

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
os.makedirs(ATTACK_DIR, exist_ok=True)
batch_size = 100

# Data
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

classes = trainloader.dataset.classes
train_size = len(trainloader.dataset)
val_size   = len(valloader.dataset)
test_size  = len(testloader.dataset)
test_inds  = np.arange(test_size)

# Model
print('==> Building model..')
if train_args['net'] == 'resnet34':
    net = ResNet34(num_classes=len(classes))
elif train_args['net'] == 'resnet101':
    net = ResNet101(num_classes=len(classes))
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

if __name__ == "__main__":

    X_val  = get_normalized_tensor(valloader, batch_size)
    y_val = valloader.dataset.targets

    X_test = get_normalized_tensor(testloader, batch_size)
    y_test = testloader.dataset.targets

    net.eval()
    classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=criterion,
                                   optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(classes))

    y_val_preds = classifier.predict(X_val, batch_size=batch_size)
    y_val_preds = y_val_preds.argmax(axis=1)
    val_acc = np.sum(y_val_preds == y_val) / val_size
    print('Accuracy on benign val examples: {}%'.format(val_acc * 100))
    np.save(os.path.join(ATTACK_DIR, 'y_val_preds.npy'), y_val_preds)

    y_test_preds = classifier.predict(X_test, batch_size=batch_size)
    y_test_preds = y_test_preds.argmax(axis=1)
    test_acc = np.sum(y_test_preds == y_test) / test_size
    print('Accuracy on benign test examples: {}%'.format(test_acc * 100))
    np.save(os.path.join(ATTACK_DIR, 'y_test_preds.npy'), y_test_preds)

    # attack
    # creating targeted labels
    if args.targeted:
        tgt_file = os.path.join(ATTACK_DIR, 'y_test_adv.npy')
        if not os.path.isfile(tgt_file):
            y_val_targets = random_targets(np.asarray(y_val), len(classes))
            y_val_adv = y_val_targets.argmax(axis=1)
            y_test_targets = random_targets(np.asarray(y_test), len(classes))
            y_test_adv = y_test_targets.argmax(axis=1)
            np.save(os.path.join(ATTACK_DIR, 'y_val_adv.npy'), y_val_adv)
            np.save(os.path.join(ATTACK_DIR, 'y_test_adv.npy'), y_test_adv)
        else:
            y_val_adv = np.load(os.path.join(ATTACK_DIR, 'y_val_adv.npy'))
            y_val_targets = to_categorical(y_val_adv, nb_classes=len(classes))
            y_test_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))
            y_test_targets = to_categorical(y_test_adv, nb_classes=len(classes))
    else:
        y_val_adv = None
        y_test_adv = None
        y_val_targets = None
        y_test_targets = None

    if args.attack == 'fgsm':
        attack = FastGradientMethod(
            classifier=classifier,
            norm=np.inf,
            eps=0.07,
            eps_step=0.003,
            targeted=args.targeted,
            num_random_init=0,
            batch_size=batch_size
        )
    elif args.attack == 'pgd':
        attack = ProjectedGradientDescent(
            classifier=classifier,
            norm=np.inf,
            eps=0.07,
            eps_step=0.003,
            targeted=args.targeted,
            batch_size=batch_size
        )
    elif args.attack == 'deepfool':
        attack = DeepFool(
            classifier=classifier,
            max_iter=50,
            epsilon=0.02,
            nb_grads=len(classes),
            batch_size=batch_size
        )
    elif args.attack == 'jsma':
        attack = SaliencyMapMethod(
            classifier,
            theta=1.0,
            gamma=0.01,
            batch_size=batch_size
        )
    elif args.attack == 'cw':
        attack = CarliniL2Method(
            classifier,
            confidence=0.8,
            targeted=args.targeted,
            initial_const=0.1,
            batch_size=batch_size
        )
    elif args.attack == 'ead':
        attack = ElasticNet(
            classifier,
            confidence=0.8,
            targeted=args.targeted,
            beta=0.01,  # EAD paper shows good results for L1
            batch_size=batch_size,
            decision_rule='L1'
        )
    else:
        err_str = 'Attack {} is not supported'.format(args.attack)
        print(err_str)
        raise AssertionError(err_str)

    X_val_adv = attack.generate(x=X_val, y=y_val_targets)
    val_adv_logits = classifier.predict(X_val_adv, batch_size=batch_size)
    y_val_adv_preds = np.argmax(val_adv_logits, axis=1)
    val_adv_accuracy = np.sum(y_val_adv_preds == y_val) / val_size

    X_test_adv = attack.generate(x=X_test, y=y_test_targets)
    test_adv_logits = classifier.predict(X_test_adv, batch_size=batch_size)
    y_test_adv_preds = np.argmax(test_adv_logits, axis=1)
    test_adv_accuracy = np.sum(y_test_adv_preds == y_test) / test_size
    print('Accuracy on adversarial val examples: {}%'.format(val_adv_accuracy * 100))
    print('Accuracy on adversarial test examples: {}%'.format(test_adv_accuracy * 100))

    # saving adv images and predictions
    np.save(os.path.join(ATTACK_DIR, 'X_val_adv.npy'), X_val_adv)
    np.save(os.path.join(ATTACK_DIR, 'y_val_adv_preds.npy'), y_val_adv_preds)
    np.save(os.path.join(ATTACK_DIR, 'X_test_adv.npy'), X_test_adv)
    np.save(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'), y_test_adv_preds)

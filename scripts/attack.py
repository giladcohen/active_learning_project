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
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.saliency_map import SaliencyMapMethod
from art.attacks.evasion.carlini import CarliniL2Method
from art.attacks.evasion.elastic_net import ElasticNet
from art.classifiers import PyTorchClassifier
from active_learning_project.attacks.zero_grad_cw_try import ZeroGrad

from cleverhans.utils import random_targets, to_categorical

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/svhn/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack', default='deepfool', type=str, help='attack: fgsm, jsma, cw, deepfool, ead, pgd')
parser.add_argument('--targeted', default=False, type=boolean_string, help='use trageted attack')
parser.add_argument('--attack_dir', default='', type=str, help='attack directory')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--subset', default=-1, type=int, help='attack only subset of test set')

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
os.makedirs(os.path.join(ATTACK_DIR, 'inds'), exist_ok=True)
batch_size = args.batch_size
rand_gen = np.random.RandomState(seed=12345)

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
    net = ResNet34(num_classes=len(classes), activation=train_args['activation'])
elif train_args['net'] == 'resnet101':
    net = ResNet101(num_classes=len(classes), activation=train_args['activation'])
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

if __name__ == "__main__":

    X_val  = get_normalized_tensor(valloader, batch_size)
    y_val  = np.asarray(valloader.dataset.targets)

    X_test = get_normalized_tensor(testloader, batch_size)
    y_test = np.asarray(testloader.dataset.targets)

    net.eval()
    classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=criterion,
                                   optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(classes))

    y_val_logits = classifier.predict(X_val, batch_size=batch_size)
    y_val_preds = y_val_logits.argmax(axis=1)
    val_acc = np.sum(y_val_preds == y_val) / val_size
    print('Accuracy on benign val examples: {}%'.format(val_acc * 100))

    y_test_logits = classifier.predict(X_test, batch_size=batch_size)
    y_test_preds = y_test_logits.argmax(axis=1)
    test_acc = np.sum(y_test_preds == y_test) / test_size
    print('Accuracy on benign test examples: {}%'.format(test_acc * 100))

    # attack
    # creating targeted labels
    if args.targeted:
        tgt_file = os.path.join(ATTACK_DIR, 'y_test_adv.npy')
        if not os.path.isfile(tgt_file):
            y_val_targets = random_targets(y_val, len(classes))
            y_val_adv = y_val_targets.argmax(axis=1)
            y_test_targets = random_targets(y_test, len(classes))
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
            estimator=classifier,
            norm=np.inf,
            eps=0.01,
            eps_step=0.001,
            targeted=args.targeted,
            num_random_init=0,
            batch_size=batch_size
        )
    elif args.attack == 'pgd':
        attack = ProjectedGradientDescent(
            estimator=classifier,
            norm=np.inf,
            eps=0.01,
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
            classifier=classifier,
            theta=1.0,
            gamma=0.01,
            batch_size=batch_size
        )
    elif args.attack == 'cw':
        attack = CarliniL2Method(
            classifier=classifier,
            confidence=0.8,
            targeted=args.targeted,
            initial_const=0.1,
            batch_size=batch_size
        )
    elif args.attack == 'ead':
        attack = ElasticNet(
            classifier=classifier,
            confidence=0.8,
            targeted=args.targeted,
            beta=0.01,  # EAD paper shows good results for L1
            batch_size=batch_size,
            decision_rule='L1'
        )
    elif args.attack == 'zga':
        attack = ZeroGrad(
            classifier=classifier,
            initial_const=0.00005,
            batch_size=100,
        )
    else:
        err_str = 'Attack {} is not supported'.format(args.attack)
        print(err_str)
        raise AssertionError(err_str)

    dump_args = args.__dict__.copy()
    dump_args['attack_params'] = {}
    for param in attack.attack_params:
        if param in attack.__dict__.keys():
            dump_args['attack_params'][param] = attack.__dict__[param]
    with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'w') as f:
        json.dump(dump_args, f, indent=2)

    # attack val set
    if args.subset == -1:  # not debug, then attack val
        if not os.path.exists(os.path.join(ATTACK_DIR, 'X_val_adv.npy')):
            X_val_adv = attack.generate(x=X_val, y=y_val_targets)
            val_adv_logits = classifier.predict(X_val_adv, batch_size=batch_size)
            y_val_adv_preds = np.argmax(val_adv_logits, axis=1)
            np.save(os.path.join(ATTACK_DIR, 'X_val_adv.npy'), X_val_adv)
            np.save(os.path.join(ATTACK_DIR, 'y_val_adv_preds.npy'), y_val_adv_preds)
        else:
            X_val_adv       = np.load(os.path.join(ATTACK_DIR, 'X_val_adv.npy'))
            y_val_adv_preds = np.load(os.path.join(ATTACK_DIR, 'y_val_adv_preds.npy'))

        val_adv_accuracy = np.mean(y_val_adv_preds == y_val)
        print('Accuracy on adversarial val examples: {}%'.format(val_adv_accuracy * 100))

    # attack test set
    if args.subset != -1:  # debug
        X_test = X_test[:args.subset]
        y_test = y_test[:args.subset]
        if y_test_targets is not None:
            y_test_targets = y_test_targets[:args.subset]

    if not os.path.exists(os.path.join(ATTACK_DIR, 'X_test_adv.npy')):
        X_test_adv = attack.generate(x=X_test, y=y_test_targets)
        test_adv_logits = classifier.predict(X_test_adv, batch_size=batch_size)
        y_test_adv_preds = np.argmax(test_adv_logits, axis=1)
        np.save(os.path.join(ATTACK_DIR, 'X_test_adv.npy'), X_test_adv)
        np.save(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'), y_test_adv_preds)
    else:
        X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
        y_test_adv_preds = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))

    test_adv_accuracy = np.mean(y_test_adv_preds == y_test)
    print('Accuracy on adversarial test examples: {}% (subset={})'.format(test_adv_accuracy * 100, args.subset))

    # dividing the official test set to a val set and to a test set
    y_test_adv_preds = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))
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
    all_inds = np.arange(test_size)

    print("Number of test samples: {}. #net_succ: {}. #net_succ_attack_flip: {}. # net_succ_attack_succ: {}"
          .format(test_size, len(f1_inds), len(f2_inds), len(f3_inds)))

    val_inds = rand_gen.choice(all_inds, int(0.5 * test_size), replace=False)
    val_inds.sort()
    f0_inds_val = np.asarray([ind for ind in f0_inds if ind in val_inds])
    f1_inds_val = np.asarray([ind for ind in f1_inds if ind in val_inds])
    f2_inds_val = np.asarray([ind for ind in f2_inds if ind in val_inds])
    f3_inds_val = np.asarray([ind for ind in f3_inds if ind in val_inds])

    test_inds = np.asarray([ind for ind in all_inds if ind not in val_inds])
    f0_inds_test = np.asarray([ind for ind in f0_inds if ind in test_inds])
    f1_inds_test = np.asarray([ind for ind in f1_inds if ind in test_inds])
    f2_inds_test = np.asarray([ind for ind in f2_inds if ind in test_inds])
    f3_inds_test = np.asarray([ind for ind in f3_inds if ind in test_inds])

    # mini_test_inds = rand_gen.choice(test_inds, 1000, replace=False)

    np.save(os.path.join(ATTACK_DIR, 'inds', 'val_inds.npy'), val_inds)
    np.save(os.path.join(ATTACK_DIR, 'inds', 'f0_inds_val.npy'), f0_inds_val)
    np.save(os.path.join(ATTACK_DIR, 'inds', 'f1_inds_val.npy'), f1_inds_val)
    np.save(os.path.join(ATTACK_DIR, 'inds', 'f2_inds_val.npy'), f2_inds_val)
    np.save(os.path.join(ATTACK_DIR, 'inds', 'f3_inds_val.npy'), f3_inds_val)

    np.save(os.path.join(ATTACK_DIR, 'inds', 'test_inds.npy'), test_inds)
    np.save(os.path.join(ATTACK_DIR, 'inds', 'f0_inds_test.npy'), f0_inds_test)
    np.save(os.path.join(ATTACK_DIR, 'inds', 'f1_inds_test.npy'), f1_inds_test)
    np.save(os.path.join(ATTACK_DIR, 'inds', 'f2_inds_test.npy'), f2_inds_test)
    np.save(os.path.join(ATTACK_DIR, 'inds', 'f3_inds_test.npy'), f3_inds_test)

    # np.save(os.path.join(ATTACK_DIR, 'inds', 'mini_test_inds.npy'), mini_test_inds)

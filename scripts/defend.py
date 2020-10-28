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
from datetime import datetime
sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from active_learning_project.utils import convert_tensor_to_image
from active_learning_project.utils import boolean_string, get_ensemble_paths
from active_learning_project.attacks.zero_grad_cw_try import ZeroGrad
from active_learning_project.classifiers.pytorch_ext_classifier import PyTorchExtClassifier

import matplotlib.pyplot as plt

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.saliency_map import SaliencyMapMethod
from art.attacks.evasion.carlini import CarliniL2Method
from art.attacks.evasion.elastic_net import ElasticNet
from art.classifiers import PyTorchClassifier


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular_softplus/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='deepfool', type=str, help='attack directory')
parser.add_argument('--rev', default='zga', type=str, help='fgsm, pgd, deepfool, none')
parser.add_argument('--rev_dir', default='debug', type=str, help='reverse dir')
parser.add_argument('--minimal', action='store_true', help='use FGSM minimal attack')
parser.add_argument('--guru', action='store_true', help='use guru labels')
parser.add_argument('--ensemble', action='store_true', help='use ensemble')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--subset', default=200, type=int, help='attack only subset of test set')

# for ZAG rev
parser.add_argument('--initial_const', default=0.01, type=float, help='guess for weight for new grad loss term')
parser.add_argument('--rev_lr', default=0.01, type=float, help='rev learning rate')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

# # DEBUG:
# args.checkpoint_dir = '/data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00'
# args.attack = ''
# args.targeted = False
# args.attack_dir = 'deepfool'
# args.rev = ''
# args.minimal = False
# args.rev_dir = ''
# args.guru = False
# args.ensemble = True
# args.ensemble_dir = '/data/gilad/logs/adv_robustness/cifar10/resnet34'

if args.rev not in ['fgsm', 'pgd', 'jsma', 'cw', 'ead']:
    assert not args.guru
if args.minimal:
    assert args.rev == 'fgsm'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)

CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

if args.rev_dir != '':
    assert args.rev != ''
    REV_DIR = os.path.join(ATTACK_DIR, 'rev', args.rev_dir)
else:
    REV_DIR = None  # no rev!
if REV_DIR is not None:
    os.makedirs(REV_DIR, exist_ok=True)

ENSEMBLE_DIR = os.path.dirname(args.checkpoint_dir)

ENSEMBLE_DIR_DUMP = os.path.join(ATTACK_DIR, 'ensemble')
os.makedirs(ENSEMBLE_DIR_DUMP, exist_ok=True)

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

X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
if targeted:
    y_test_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))

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

# get and assert preds:
net.eval()
classifier = PyTorchExtClassifier(model=net, clip_values=(0, 1), loss=criterion,
                                  optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(classes))

y_test_logits = classifier.predict(X_test, batch_size=batch_size)
y_test_preds = y_test_logits.argmax(axis=1)
try:
    np.testing.assert_array_almost_equal_nulp(y_test_preds, np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy')))
    np.save(os.path.join(ATTACK_DIR, 'y_test_logits.npy'), y_test_logits)
except AssertionError as e:
    raise AssertionError('{}\nAssert failed for y_test_preds for ATTACK_DIR={}'.format(e, ATTACK_DIR))

y_test_adv_logits = classifier.predict(X_test_adv, batch_size=batch_size)
y_test_adv_preds = y_test_adv_logits.argmax(axis=1)
try:
    np.testing.assert_array_almost_equal_nulp(y_test_adv_preds, np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy')))
    np.save(os.path.join(ATTACK_DIR, 'y_test_adv_logits.npy'), y_test_adv_logits)
except AssertionError as e:
    raise AssertionError('{}\nAssert failed for y_test_adv_logits for ATTACK_DIR={}'.format(e, ATTACK_DIR))

subset = attack_args.get('subset', -1)  # default to attack
# may be overriden:
if args.subset != -1:
    subset = args.subset

if subset != -1:  # if debug run
    # X_test = X_test[:subset]
    # y_test = y_test[:subset]
    # X_test_adv = X_test_adv[:subset]
    # if targeted:
    #     y_test_adv = y_test_adv[:subset]
    # y_test_logits = y_test_logits[:subset]
    # y_test_preds = y_test_preds[:subset]
    # y_test_adv_logits = y_test_adv_logits[:subset]
    # y_test_adv_preds = y_test_adv_preds[:subset]
    test_size = subset

# what are the samples we care about? net_succ (not attack_succ. it is irrelevant)
f0_inds = []  # net_fail
f1_inds = []  # net_succ
f2_inds = []  # net_succ AND attack_flip
f3_inds = []  # net_succ AND attack_flip AND attack_succ

for i in range(test_size):
    f1 = y_test_preds[i] == y_test[i]
    f2 = f1 and y_test_preds[i] != y_test_adv_preds[i]
    if targeted:
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

print("Number of test samples: {}. #net_succ: {}. #net_succ_attack_flip: {}. # net_succ_attack_succ: {}"
      .format(test_size, len(f1_inds), len(f2_inds), len(f3_inds)))

# reverse attack:
if args.rev == 'fgsm':
    defense = FastGradientMethod(
        estimator=classifier,
        norm=np.inf,
        eps=0.01,
        eps_step=0.001,
        targeted=args.guru,
        num_random_init=0,
        batch_size=batch_size,
        minimal=args.minimal
    )
elif args.rev == 'pgd':
    defense = ProjectedGradientDescent(
        estimator=classifier,
        norm=np.inf,
        eps=0.01,
        eps_step=0.003,
        targeted=args.guru,
        batch_size=batch_size
    )
elif args.rev == 'deepfool':
    defense = DeepFool(
        classifier=classifier,
        epsilon=0.02,
        nb_grads=len(classes),
        batch_size=batch_size
    )
elif args.rev == 'jsma':
    defense = SaliencyMapMethod(
        classifier=classifier,
        theta=1.0,
        gamma=0.01,
        batch_size=batch_size
    )
elif args.rev == 'cw':
    defense = CarliniL2Method(
        classifier=classifier,
        confidence=0.8,
        targeted=args.guru,
        initial_const=0.1,
        batch_size=batch_size
    )
elif args.rev == 'ead':
    defense = ElasticNet(
        classifier=classifier,
        confidence=0.8,
        targeted=args.guru,
        beta=0.01,  # EAD paper shows good results for L1
        batch_size=batch_size,
        decision_rule='L1'
    )
elif args.rev == 'zga':
    defense = ZeroGrad(
        classifier=classifier,
        initial_const=args.initial_const,
        learning_rate=args.rev_lr,
        batch_size=100,
    )
elif args.rev == '':
    assert args.ensemble
    defense = None
else:
    raise AssertionError('Unknown rev {}'.format(args.rev))

dump_args = args.__dict__.copy()
if defense is not None:
    dump_args['defense_params'] = {}
    for param in defense.attack_params:
        dump_args['defense_params'][param] = defense.__dict__[param]
    with open(os.path.join(REV_DIR, 'defense_args.txt'), 'w') as f:
        json.dump(dump_args, f, indent=2)

if args.guru:
    y_targets = y_test[:test_size]  # is used for the defense, thus, always subset
else:
    y_targets = None

if defense:  # if rev is not empty
    # if necessary, generate the rev images
    if not os.path.exists(os.path.join(REV_DIR, 'X_test_rev.npy')):
        print('main normal rev images were not calculated before. Generating...')
        X_test_rev = defense.generate(x=X_test[:test_size], y=y_targets)
        np.save(os.path.join(REV_DIR, 'X_test_rev.npy'), X_test_rev)
    else:
        print('main normal rev images were already calculated before. Loading...')
        X_test_rev = np.load(os.path.join(REV_DIR, 'X_test_rev.npy'))

    if not os.path.exists(os.path.join(REV_DIR, 'X_test_adv_rev.npy')):
        print('main adv rev images were not calculated before. Generating...')
        X_test_adv_rev = defense.generate(x=X_test_adv[:test_size], y=y_targets)
        np.save(os.path.join(REV_DIR, 'X_test_adv_rev.npy'), X_test_adv_rev)
    else:
        print('main adv rev images were already calculated before. Loading...')
        X_test_adv_rev = np.load(os.path.join(REV_DIR, 'X_test_adv_rev.npy'))

    y_test_rev_logits = classifier.predict(X_test_rev, batch_size=batch_size)
    y_test_rev_preds = y_test_rev_logits.argmax(axis=1)
    np.save(os.path.join(REV_DIR, 'y_test_rev_logits.npy'), y_test_rev_logits)
    np.save(os.path.join(REV_DIR, 'y_test_rev_preds.npy'), y_test_rev_preds)

    y_test_adv_rev_logits = classifier.predict(X_test_adv_rev, batch_size=batch_size)
    y_test_adv_rev_preds = y_test_adv_rev_logits.argmax(axis=1)
    np.save(os.path.join(REV_DIR, 'y_test_adv_rev_logits.npy'), y_test_adv_rev_logits)
    np.save(os.path.join(REV_DIR, 'y_test_adv_rev_preds.npy'), y_test_adv_rev_preds)

if args.ensemble:
    print('Running ensemble defense. Loading all models')
    ensemble_paths = get_ensemble_paths(ENSEMBLE_DIR)
    ensemble_paths = ensemble_paths[1:]  # ignoring the first (original) network

    # calculated all the time, even without rev. For normal:
    y_test_net_logits_mat = np.empty((X_test.shape[0], len(ensemble_paths), len(classes)), dtype=np.float32)  # (N, #nets, #classes)
    y_test_net_preds_mat = np.empty((X_test.shape[0], len(ensemble_paths)), dtype=np.int32)  # (N, #nets)

    # and for adv:
    y_test_adv_net_logits_mat = np.empty_like(y_test_net_logits_mat) # (N, #nets, #classes)
    y_test_adv_net_preds_mat = np.empty_like(y_test_net_preds_mat)  # (N, #nets)

    if defense:  # calculated only for rev:
        normal_rev_exist = os.path.exists(os.path.join(REV_DIR, 'X_test_rev_mat.npy'))
        adv_rev_exist    = os.path.exists(os.path.join(REV_DIR, 'X_test_adv_rev_mat.npy'))

        if not normal_rev_exist:  # will be calculated. Very time consuming
            print('ensemble normal rev images were not calculated before. Initializing mat...')
            X_test_rev_mat = np.empty((test_size, len(ensemble_paths)) + X_test.shape[1:], dtype=np.float32)
        else:
            print('ensemble normal rev images were already calculated before. Loading...')
            X_test_rev_mat = np.load(os.path.join(REV_DIR, 'X_test_rev_mat.npy'))

        if not adv_rev_exist:  # will be calculated. Very time consuming
            print('ensemble adv rev images were not calculated before. Initializing mat...')
            X_test_adv_rev_mat = np.empty_like(X_test_rev_mat)
        else:
            print('ensemble adv rev images were already calculated before. Loading...')
            X_test_adv_rev_mat = np.load(os.path.join(REV_DIR, 'X_test_adv_rev_mat.npy'))

        y_test_net_rev_logits_mat = np.empty((test_size, len(ensemble_paths), len(classes)), dtype=np.float32)  # (N, #nets, #classes)
        y_test_net_rev_preds_mat = np.empty((test_size, len(ensemble_paths)), dtype=np.int32)  # (N, #nets)

        y_test_adv_net_rev_logits_mat = np.empty_like(y_test_net_rev_logits_mat)  # (N, #nets, #classes)
        y_test_adv_net_rev_preds_mat = np.empty_like(y_test_net_rev_preds_mat)  # (N, #nets)

    for i, ckpt_file in tqdm(enumerate(ensemble_paths)):
        global_state = torch.load(ckpt_file, map_location=torch.device(device))
        net.load_state_dict(global_state['best_net'])
        print('{}: fetching predictions using ckpt file: {}'.format(datetime.now().strftime("%H:%M:%S"), ckpt_file))

        y_test_net_logits_mat[:, i] = classifier.predict(X_test, batch_size=batch_size)
        y_test_net_preds_mat[:, i] = y_test_net_logits_mat[:, i].argmax(axis=1)

        y_test_adv_net_logits_mat[:, i] = classifier.predict(X_test_adv, batch_size=batch_size)
        y_test_adv_net_preds_mat[:, i] = y_test_adv_net_logits_mat[:, i].argmax(axis=1)

        # probability for same prediction for normal images:
        prob_all = np.mean(y_test_net_preds_mat[:, i] == y_test_preds)
        prob_f1 = np.mean(y_test_net_preds_mat[:, i][f1_inds] == y_test_preds[f1_inds])
        prob_f2 = np.mean(y_test_net_preds_mat[:, i][f2_inds] == y_test_preds[f2_inds])
        prob_f3 = np.mean(y_test_net_preds_mat[:, i][f3_inds] == y_test_preds[f3_inds])
        print('Probability of normal net prediction to match the normal prediction: all samples: {:.2f}%. f1 samples: {:.2f}%, f2 samples: {:.2f}%, f3 samples: {:.2f}%'
              .format(prob_all * 100, prob_f1 * 100, prob_f2 * 100, prob_f3 * 100))

        # probability for same prediction for adv images:
        prob_all = np.mean(y_test_adv_net_preds_mat[:, i] == y_test_adv_preds)
        prob_f1 = np.mean(y_test_adv_net_preds_mat[:, i][f1_inds] == y_test_adv_preds[f1_inds])
        prob_f2 = np.mean(y_test_adv_net_preds_mat[:, i][f2_inds] == y_test_adv_preds[f2_inds])
        prob_f3 = np.mean(y_test_adv_net_preds_mat[:, i][f3_inds] == y_test_adv_preds[f3_inds])
        print('Probability of adv net prediction to match the adv prediction: all samples: {:.2f}%. f1 samples: {:.2f}%, f2 samples: {:.2f}%, f3 samples: {:.2f}%'
              .format(prob_all * 100, prob_f1 * 100, prob_f2 * 100, prob_f3 * 100))

        if defense:
            if not normal_rev_exist:
                print('Generating normal rev images for network {}'.format(ckpt_file))
                X_test_rev_mat[:, i] = defense.generate(x=X_test[:test_size], y=y_targets)

            if not adv_rev_exist:
                print('Generating adv rev images for network {}'.format(ckpt_file))
                X_test_adv_rev_mat[:, i] = defense.generate(x=X_test_adv[:test_size], y=y_targets)

            # normal preds for net rev:
            y_test_net_rev_logits_mat[:, i] = classifier.predict(X_test_rev_mat[:, i], batch_size=batch_size)
            y_test_net_rev_preds_mat[:, i] = y_test_net_rev_logits_mat[:, i].argmax(axis=1)

            # adv preds for net rev:
            y_test_adv_net_rev_logits_mat[:, i] = classifier.predict(X_test_adv_rev_mat[:, i], batch_size=batch_size)
            y_test_adv_net_rev_preds_mat[:, i] = y_test_adv_net_rev_logits_mat[:, i].argmax(axis=1)

    np.save(os.path.join(ENSEMBLE_DIR_DUMP, 'y_test_net_logits_mat.npy'), y_test_net_logits_mat)
    np.save(os.path.join(ENSEMBLE_DIR_DUMP, 'y_test_net_preds_mat.npy'), y_test_net_preds_mat)
    np.save(os.path.join(ENSEMBLE_DIR_DUMP, 'y_test_adv_net_logits_mat.npy'), y_test_adv_net_logits_mat)
    np.save(os.path.join(ENSEMBLE_DIR_DUMP, 'y_test_adv_net_preds_mat.npy'), y_test_adv_net_preds_mat)

    if defense:
        if not normal_rev_exist:
            np.save(os.path.join(REV_DIR, 'X_test_rev_mat.npy'), X_test_rev_mat)
        if not adv_rev_exist:
            np.save(os.path.join(REV_DIR, 'X_test_adv_rev_mat.npy'), X_test_adv_rev_mat)

        # dumping normal preds for net rev
        np.save(os.path.join(REV_DIR, 'y_test_net_rev_logits_mat.npy'), y_test_net_rev_logits_mat)
        np.save(os.path.join(REV_DIR, 'y_test_net_rev_preds_mat.npy'), y_test_net_rev_preds_mat)

        # dumping adv preds for net rev
        np.save(os.path.join(REV_DIR, 'y_test_adv_net_rev_logits_mat.npy'), y_test_adv_net_rev_logits_mat)
        np.save(os.path.join(REV_DIR, 'y_test_adv_net_rev_preds_mat.npy'), y_test_adv_net_rev_preds_mat)

print('done')

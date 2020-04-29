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
from active_learning_project.utils import boolean_string, majority_vote

import matplotlib.pyplot as plt

from art.attacks import FastGradientMethod, ProjectedGradientDescent, DeepFool, SaliencyMapMethod, CarliniL2Method, \
    ElasticNet
from art.classifiers import PyTorchClassifier

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack', default='fgsm', type=str, help='checkpoint dir')
parser.add_argument('--targeted', default=True, type=boolean_string, help='use targeted attack')
parser.add_argument('--minimal', action='store_true', help='use FGSM minimal attack')
parser.add_argument('--attack_dir', default='', type=str, help='attack directory')
parser.add_argument('--rev', default='pgd', type=str, help='fgsm, pgd, deepfool, none')
parser.add_argument('--rev_dir', default='', type=str, help='reverse dir')
parser.add_argument('--guru', action='store_true', help='use guru labels')
parser.add_argument('--ensemble', action='store_true', help='use ensemble')
parser.add_argument('--ensemble_dir', default='', type=str, help='ensemble dir of many networks')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

# DEBUG:
# args.checkpoint_dir = '/data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00'
# args.attack = 'cw'
# args.targeted = True
# args.rev = 'fgsm'
# args.rev_dir = 'debug'
# args.guru = False
# args.ensemble = False
# args.ensemble_dir = '/data/gilad/logs/adv_robustness/cifar10/resnet34'

if args.rev not in ['fgsm', 'pgd', 'jsma', 'cw', 'ead']:
    assert not args.guru
if args.minimal:
    assert args.rev == 'fgsm'

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
if args.rev_dir != '':
    REV_DIR = os.path.join(ATTACK_DIR, 'rev', args.rev_dir)
else:
    REV_DIR = os.path.join(ATTACK_DIR, 'rev', args.rev)
os.makedirs(REV_DIR, exist_ok=True)
if args.ensemble_dir != '':
    ENSEMBLE_DIR = args.ensemble_dir
else:
    ENSEMBLE_DIR = os.path.dirname(args.checkpoint_dir)

batch_size = 100

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
if args.targeted:
    y_test_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))

# Model
print('==> Building model..')
if train_args['net'] == 'resnet34':
    net = ResNet34(num_classes=len(classes))
elif train_args['net'] == 'resnet101':
    net = ResNet101(num_classes=len(classes))
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
classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=criterion,
                               optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(classes))

y_test_preds = classifier.predict(X_test, batch_size=batch_size).argmax(axis=1)
assert (y_test_preds == np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy'))).all()

y_test_adv_preds = classifier.predict(X_test_adv, batch_size=batch_size).argmax(axis=1)
assert (y_test_adv_preds == np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))).all()

# what are the samples we care about? net_succ and attack_succ only
f_inds = []
for i in range(test_size):
    collect = (y_test_preds[i] == y_test[i]) and \
              (y_test_preds[i] != y_test_adv_preds[i])
    if args.targeted:
        collect = collect and y_test_adv_preds[i] == y_test_adv[i]
    if collect:
        f_inds.append(i)
f_inds = np.asarray(f_inds)

# reverse attack:
if args.rev == 'fgsm':
    defense = FastGradientMethod(
        classifier=classifier,
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
        classifier=classifier,
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
        classifier,
        theta=1.0,
        gamma=0.01,
        batch_size=batch_size
    )
elif args.rev == 'cw':
    defense = CarliniL2Method(
        classifier,
        confidence=0.8,
        targeted=args.guru,
        initial_const=0.1,
        batch_size=batch_size
    )
elif args.rev == 'ead':
    defense = ElasticNet(
        classifier,
        confidence=0.8,
        targeted=args.guru,
        beta=0.01,  # EAD paper shows good results for L1
        batch_size=batch_size,
        decision_rule='L1'
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
    y_targets = y_test
else:
    y_targets = None

if not args.ensemble:
    X_test_rev = defense.generate(x=X_test_adv, y=y_targets)
    y_test_rev_preds = classifier.predict(X_test_rev, batch_size=batch_size).argmax(axis=1)
    np.save(os.path.join(REV_DIR, 'X_test_rev.npy'), X_test_rev)
    np.save(os.path.join(REV_DIR, 'y_test_rev_preds.npy'), y_test_rev_preds)
else:  # use ensemble
    print('Running ensemble defense. Loading all models')
    checkpoint_dir_list = next(os.walk(ENSEMBLE_DIR))[1]
    checkpoint_dir_list.sort()
    checkpoint_dir_list = checkpoint_dir_list[1:]  # ignoring the first (original) network
    y_test_pred_mat = np.empty((test_size, len(checkpoint_dir_list)), dtype=np.int32)
    y_test_pred_mat_orig = np.empty_like(y_test_pred_mat)  # for debug

    for i, dir in tqdm(enumerate(checkpoint_dir_list)):
        ckpt_file = os.path.join(ENSEMBLE_DIR, dir, 'ckpt.pth')
        global_state = torch.load(ckpt_file, map_location=torch.device(device))
        net.load_state_dict(global_state['best_net'])
        print('fetching predictions using ckpt file: {}'.format(ckpt_file))
        y_test_preds_tmp = classifier.predict(X_test_adv, batch_size=batch_size).argmax(axis=1)
        y_test_pred_mat_orig[:, i] = y_test_preds_tmp.copy()  # debug

        if args.rev != '':
            acc_all = np.mean(y_test_preds_tmp == y_test_adv_preds)
            acc_filtered = np.mean(y_test_preds_tmp[f_inds] == y_test_adv_preds[f_inds])
            print('accuracy of net prediction to the adversarial predictions: all samples: {:.2f}%. filtered samples: {:.2f}%'
                  .format(acc_all * 100, acc_filtered * 100))

            #If a label is just like y_test_adv_preds, take the prediction label after an attack (if it is different)')
            X_test_rev_tmp = defense.generate(x=X_test_adv, y=y_targets)
            y_test_rev_preds = classifier.predict(X_test_rev_tmp, batch_size=batch_size).argmax(axis=1)

            # the below counts are relevant only for net_succ and attack_succ only
            rev_succ_switch_cnt = 0
            rev_fail_switch_cnt = 0
            nan_switch_cnt = 0
            for k in range(test_size):
                # collect only for net_succ and attack_succ
                collect = k in f_inds

                if y_test_preds_tmp[k] == y_test_adv_preds[k]:  # if we have the same prediction as the adversarial prediction
                    if y_test_preds_tmp[k] != y_test_rev_preds[k]:  # if the rev defense label switched the adv label
                        y_test_preds_tmp[k] = y_test_rev_preds[k]
                        if collect:
                            if y_test_rev_preds[k] == y_test[k]:  # if rev label match the GT
                                rev_succ_switch_cnt += 1
                            else:
                                rev_fail_switch_cnt += 1
                    else:  # if the rev defense label didn't manage to switch the the adv label
                        y_test_preds_tmp[k] = -1
                        if collect:
                            nan_switch_cnt += 1

            print('For filtered samples: overall switched: {}. correct: {}, incorrect: {}, unknown: {}'
                  .format(rev_succ_switch_cnt + rev_fail_switch_cnt + nan_switch_cnt, rev_succ_switch_cnt, rev_fail_switch_cnt, nan_switch_cnt))

        y_test_pred_mat[:, i] = y_test_preds_tmp

    y_test_rev_preds = np.apply_along_axis(majority_vote, axis=1, arr=y_test_pred_mat)
    np.save(os.path.join(REV_DIR, 'y_test_rev_preds.npy'), y_test_rev_preds)
    np.save(os.path.join(REV_DIR, 'y_test_pred_mat_orig.npy'), y_test_pred_mat_orig)
    np.save(os.path.join(REV_DIR, 'y_test_pred_mat.npy'), y_test_pred_mat)

# DEBUG:
# i = 9999
# strr = 'class is {}({}), model predicted {}({}), '.format(classes[y_test[i]], y_test[i], classes[y_test_preds[i]], y_test_preds[i])
# if args.targeted:
#     strr += 'we wanted to attack to {}({}), '.format(classes[y_test_adv[i]], y_test_adv[i])
# strr += 'and after adv noise: {}({}).\n'.format(classes[y_test_adv_preds[i]], y_test_adv_preds[i])
# strr += 'original ensemble predictions: {}\n'.format(y_test_pred_mat_orig[i])
# strr += 'reverted ensemble predictions: {}\n'.format(y_test_pred_mat[i])
# print(strr)


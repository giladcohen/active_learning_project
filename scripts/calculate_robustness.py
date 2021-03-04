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
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.utils import boolean_string, majority_vote, get_ensemble_paths, add_feature
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor
from scipy.special import softmax
from cleverhans.utils import to_categorical, batch_indices

import matplotlib.pyplot as plt
from art.classifiers import PyTorchClassifier
# from active_learning_project.classifiers.pytorch_ext_classifier import PyTorchExtClassifier

parser = argparse.ArgumentParser(description='Adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/Users/giladcohen/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='ead', type=str, help='attack directory')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--net_pool', default='main', type=str, help='networks pool: main, ensemble, all')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

ENSEMBLE_DIR = os.path.dirname(args.checkpoint_dir)

batch_size = args.batch_size
rand_gen = np.random.RandomState(seed=12345)

# Data
print('==> Preparing data..')
testloader = get_test_loader(
    dataset=train_args['dataset'],
    batch_size=batch_size,
    num_workers=1,
    pin_memory=True
)

classes = testloader.dataset.classes
test_size = len(testloader.dataset)
test_inds = np.arange(test_size)

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

num_batches = int(np.ceil(test_size/batch_size))

# what are the samples we care about? net_succ (not attack_succ. it is irrelevant)
# load inds
val_inds     = np.load(os.path.join(ATTACK_DIR, 'inds', 'val_inds.npy'))
f0_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f0_inds_val.npy'))
f1_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f1_inds_val.npy'))
f2_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f2_inds_val.npy'))
f3_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f3_inds_val.npy'))

test_inds    = np.load(os.path.join(ATTACK_DIR, 'inds', 'test_inds.npy'))
f0_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f0_inds_test.npy'))
f1_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f1_inds_test.npy'))
f2_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f2_inds_test.npy'))
f3_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f3_inds_test.npy'))

print("Number of test samples: {}. #net_succ: {}. #net_succ_attack_flip: {}. #net_succ_attack_succ: {}"
      .format(len(test_inds), len(f1_inds_test), len(f2_inds_test), len(f3_inds_test)))

networks_list = []
if args.net_pool in ['main', 'all']:
    networks_list.append(CHECKPOINT_PATH)
if args.net_pool in ['ensemble', 'all']:
    networks_list.extend(get_ensemble_paths(ENSEMBLE_DIR))
    networks_list.remove(CHECKPOINT_PATH)
num_networks = len(networks_list)

y_preds_nets     = -1 * np.ones((test_size, num_networks), dtype=np.int32)
y_preds_nets_adv = -1 * np.ones((test_size, num_networks), dtype=np.int32)

for j, ckpt_file in tqdm(enumerate(networks_list)):  # for network j
    print('Evaluating network {}'.format(ckpt_file))
    global_state = torch.load(ckpt_file, map_location=torch.device(device))
    net.load_state_dict(global_state['best_net'])
    y_preds_nets[:, j]     = classifier.predict(X_test, batch_size=batch_size).argmax(axis=1)
    y_preds_nets_adv[:, j] = classifier.predict(X_test_adv, batch_size=batch_size).argmax(axis=1)

# use majority vote:
y_preds     = np.apply_along_axis(majority_vote, axis=1, arr=y_preds_nets)
y_preds_adv = np.apply_along_axis(majority_vote, axis=1, arr=y_preds_nets_adv)

def calc_robust_metrics(robustness_preds, robustness_preds_adv):
    acc_all = np.mean(robustness_preds[test_inds] == y_test[test_inds])
    acc_f1 = np.mean(robustness_preds[f1_inds_test] == y_test[f1_inds_test])
    acc_f2 = np.mean(robustness_preds[f2_inds_test] == y_test[f2_inds_test])
    acc_f3 = np.mean(robustness_preds[f3_inds_test] == y_test[f3_inds_test])

    acc_all_adv = np.mean(robustness_preds_adv[test_inds] == y_test[test_inds])
    acc_f1_adv = np.mean(robustness_preds_adv[f1_inds_test] == y_test[f1_inds_test])
    acc_f2_adv = np.mean(robustness_preds_adv[f2_inds_test] == y_test[f2_inds_test])
    acc_f3_adv = np.mean(robustness_preds_adv[f3_inds_test] == y_test[f3_inds_test])

    print('Robust classification accuracy: all samples: {:.2f}/{:.2f}%, f1 samples: {:.2f}/{:.2f}%, f2 samples: {:.2f}/{:.2f}%, f3 samples: {:.2f}/{:.2f}%'
          .format(acc_all * 100, acc_all_adv * 100, acc_f1 * 100, acc_f1_adv * 100, acc_f2 * 100, acc_f2_adv * 100, acc_f3 * 100, acc_f3_adv * 100))

calc_robust_metrics(y_preds, y_preds_adv)

"""
This script uses TTA and random forest for robustness and adv detection.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
import torchvision.transforms as transforms

import logging
import numpy as np
from numba import njit
import json
import os
import argparse
from tqdm import tqdm
import time
import sys
import PIL
from sklearn.ensemble import RandomForestClassifier

from torchlars import LARS
import matplotlib.pyplot as plt

sys.path.insert(0, "..")

import active_learning_project.datasets.my_transforms as my_transforms
from active_learning_project.datasets.my_cifar10_ttas import TTADataset
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor, \
    get_single_img_dataloader, get_explicit_train_loader
from active_learning_project.utils import EMA, update_moving_average, convert_tensor_to_image, set_logger, get_model, \
    reset_net, pytorch_evaluate
from active_learning_project.datasets.utils import get_mini_dataset_inds
from active_learning_project.metric_utils import calc_first_n_adv_acc, calc_first_n_adv_acc_from_probs_summation
from Pointnet_Pointnet2_pytorch.models.pointnet_utils import feature_transform_reguliarzer


parser = argparse.ArgumentParser(description='PyTorch TTA defense with mlp')
parser.add_argument('--checkpoint_dir',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00',
                    type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='cw_targeted', type=str, help='attack directory')

# eval:
parser.add_argument('--tta_size', default=1000, type=int, help='number of test-time augmentations')
parser.add_argument('--features', default='probs', type=str, help='which features to use from resnet: embeddings/logits/probs')
parser.add_argument('--num_workers', default=20, type=int, help='number of workers')

# transforms:
parser.add_argument('--clip_inputs', action='store_true', help='clipping TTA inputs between 0 and 1')
parser.add_argument('--gaussian_std', default=0.0, type=float, help='Standard deviation of Gaussian noise')  # was 0.0125
parser.add_argument('--soft_transforms', action='store_true', help='applying mellow transforms')

# RF training:
parser.add_argument('--val_size', default=7500, type=int, help='validation size')

# debug:
parser.add_argument('--test_size', default=None, type=int, help='test size')
parser.add_argument('--dump_dir', default='tmp', type=str, help='the dump dir')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
DUMP_DIR = os.path.join(ATTACK_DIR, args.dump_dir)
os.makedirs(DUMP_DIR, exist_ok=True)
log_file = os.path.join(DUMP_DIR, 'log.log')

# dumping args to txt file
with open(os.path.join(DUMP_DIR, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

set_logger(log_file)
logger = logging.getLogger()

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
dataset = train_args['dataset']
val_inds, test_inds = get_mini_dataset_inds(dataset)

with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

rand_gen = np.random.RandomState(12345)

# Data
logger.info('==> Preparing data..')
all_dataset_eval_loader = get_test_loader(
    dataset=dataset,
    batch_size=100,
    num_workers=1,
    pin_memory=True
)

st = args.soft_transforms
if args.clip_inputs == True:
    clip_min, clip_max = 0.0, 1.0
else:
    clip_min, clip_max = -np.inf, np.inf
p_hflip = 0.5 if 'cifar' in dataset else 0.0
tta_transforms = transforms.Compose([
    my_transforms.Clip(0.0, 1.0),  # TO fix a bug where an ADV image has minus small value, applying gamma yields Nan
    my_transforms.ColorJitterPro(
        brightness=[0.8, 1.2] if st else [0.6, 1.4],
        contrast=[0.85, 1.15] if st else [0.7, 1.3],
        saturation=[0.75, 1.25] if st else [0.5, 1.5],
        hue=[-0.03, 0.03] if st else [-0.06, 0.06],
        gamma=[0.85, 1.15] if st else [0.7, 1.3]
    ),
    transforms.Pad(padding=16, padding_mode='edge'),
    transforms.RandomAffine(
        degrees=[-8, 8] if st else [-15, 15],
        translate=(4.0 / 64, 4.0 / 64),
        scale=(0.95, 1.05) if st else (0.9, 1.1),
        shear=None,
        resample=PIL.Image.BILINEAR,
        fillcolor=None
    ),
    transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if st else [0.001, 0.5]),
    transforms.CenterCrop(size=32),
    transforms.RandomHorizontalFlip(p=p_hflip),
    my_transforms.GaussianNoise(0, args.gaussian_std),
    my_transforms.Clip(clip_min, clip_max)
])

global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
classes = all_dataset_eval_loader.dataset.classes

# get data for TTA loaders
X_test           = get_normalized_tensor(all_dataset_eval_loader)
y_test           = np.asarray(all_dataset_eval_loader.dataset.targets)
X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
img_shape = X_test.shape[1:]

# filter by mini_val and mini_test sets
if args.val_size is not None:
    val_inds         = val_inds[0:args.val_size]
if args.test_size is not None:
    test_inds = test_inds[0:args.test_size]
val_size = len(val_inds)
test_size = len(test_inds)

net = get_model(train_args['net'])(num_classes=len(classes), activation=train_args['activation'])
net = net.to(device)
net.load_state_dict(global_state['best_net'])
net.eval()  # frozen

if args.features in ['probs', 'logits']:
    num_channels = len(classes)
elif args.features == 'embeddings':
    num_channels = net.linear.in_features
else:
    AssertionError('Unexpected args.features={}'.format(args.features))

# summary(net, (3, 32, 32))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

tta_dataset_train = TTADataset(
    torch.from_numpy(X_test[val_inds]),
    torch.from_numpy(X_test_adv[val_inds]),
    torch.from_numpy(y_test[val_inds]),
    args.tta_size,
    transform=tta_transforms)

tta_dataset_test = TTADataset(
    torch.from_numpy(X_test[test_inds]),
    torch.from_numpy(X_test_adv[test_inds]),
    torch.from_numpy(y_test[test_inds]),
    args.tta_size,
    transform=tta_transforms)

train_loader = torch.utils.data.DataLoader(
    tta_dataset_train, batch_size=1, shuffle=True,
    num_workers=args.num_workers, pin_memory=device=='cuda')

test_loader = torch.utils.data.DataLoader(
    tta_dataset_test, batch_size=1, shuffle=False,
    num_workers=args.num_workers, pin_memory=device=='cuda')


def rearrange_as_pts(x: torch.Tensor) -> torch.Tensor:
    """Reshape the x tensor from [B * N, D] to [B, N, D]"""
    x = x.reshape(-1, args.tta_size, num_channels)
    return x


y_gt_train        = np.nan * np.ones(2 * val_size)
y_is_adv_gt_train = np.nan * np.ones(2 * val_size)
features_train    = np.nan * np.ones((2 * val_size, args.tta_size, num_channels), dtype=np.float32)
train_cnt = 0

y_gt_test        = np.nan * np.ones(2 * test_size)
y_is_adv_gt_test = np.nan * np.ones(2 * test_size)
features_test    = np.nan * np.ones((2 * test_size, args.tta_size, num_channels), dtype=np.float32)
test_cnt = 0

logger.info('collecting validation features...')
with torch.no_grad():
    for batch_idx, (x, y, y_is_adv) in tqdm(enumerate(train_loader)):
        x, y, y_is_adv = x.reshape((-1,) + img_shape), y.reshape(-1), y_is_adv.reshape(-1)
        x, y, y_is_adv = x.to(device), y.to(device), y_is_adv.to(device)
        b = train_cnt
        e = b + y.size(0)
        features_train[b:e] = rearrange_as_pts(net(x)[args.features]).cpu().numpy()
        y_gt_train[b:e] = y.cpu().numpy()
        y_is_adv_gt_train[b:e] = y_is_adv.cpu().numpy()
        train_cnt += y.size(0)
assert train_cnt == 2 * val_size
assert not np.isnan(features_train).any()
assert not np.isnan(y_gt_train).any()
assert not np.isnan(y_is_adv_gt_train).any()

logger.info('collecting test features...')
with torch.no_grad():
    for batch_idx, (x, y, y_is_adv) in tqdm(enumerate(test_loader)):
        x, y, y_is_adv = x.reshape((-1,) + img_shape), y.reshape(-1), y_is_adv.reshape(-1)
        x, y, y_is_adv = x.to(device), y.to(device), y_is_adv.to(device)
        b = test_cnt
        e = b + y.size(0)
        features_test[b:e] = rearrange_as_pts(net(x)[args.features]).cpu().numpy()
        y_gt_test[b:e] = y.cpu().numpy()
        y_is_adv_gt_test[b:e] = y_is_adv.cpu().numpy()
        test_cnt += y.size(0)
assert test_cnt == 2 * test_size
assert not np.isnan(features_test).any()
assert not np.isnan(y_gt_test).any()
assert not np.isnan(y_is_adv_gt_test).any()

# masking
train_normal_indices = np.where(y_is_adv_gt_train == 0)[0]
train_adv_indices = np.where(y_is_adv_gt_train == 1)[0]
test_normal_indices = np.where(y_is_adv_gt_test == 0)[0]
test_adv_indices = np.where(y_is_adv_gt_test == 1)[0]

# training is_adv classifier
features_train = features_train.reshape((2 * val_size, args.tta_size * num_channels))
features_test = features_test.reshape((2 * test_size, args.tta_size * num_channels))
is_adv_rf = RandomForestClassifier(
    n_estimators=1000,
    criterion="entropy",  # gini or entropy
    max_depth=None, # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
    # until all leaves contain less than min_samples_split samples.
    min_samples_split=10,
    min_samples_leaf=10,
    bootstrap=True, # Whether bootstrap samples are used when building trees.
    # If False, the whole datset is used to build each tree.
    random_state=rand_gen,
    verbose=1000,
    n_jobs=20
)
is_adv_rf.fit(features_train, y_is_adv_gt_train)
y_is_adv_preds = is_adv_rf.predict(features_test)
is_adv_acc = np.mean(y_is_adv_preds == y_is_adv_gt_test)
is_adv_acc_normal = np.mean(y_is_adv_preds[test_normal_indices] == y_is_adv_gt_test[test_normal_indices])
is_adv_acc_adv = np.mean(y_is_adv_preds[test_adv_indices] == y_is_adv_gt_test[test_adv_indices])

# training label robustness classifier
cls_rf = RandomForestClassifier(
    n_estimators=1000,
    criterion="entropy",  # gini or entropy
    max_depth=None, # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
    # until all leaves contain less than min_samples_split samples.
    min_samples_split=10,
    min_samples_leaf=10,
    bootstrap=True, # Whether bootstrap samples are used when building trees.
    # If False, the whole datset is used to build each tree.
    random_state=rand_gen,
    verbose=1000,
    n_jobs=20
)
cls_rf.fit(features_train, y_gt_train)
y_preds = cls_rf.predict(features_test)
cls_acc = np.mean(y_preds == y_gt_test)
cls_acc_normal = np.mean(y_preds[test_normal_indices] == y_gt_test[test_normal_indices])
cls_acc_adv = np.mean(y_preds[test_adv_indices] == y_gt_test[test_adv_indices])

# logging results:
logger.info('is_adv classification: acc={:.4f}, normal_acc={:.4f}, adv_acc={:.4f}'
            .format(100.0 * is_adv_acc, 100.0 * is_adv_acc_normal, 100.0 * is_adv_acc_adv))

logger.info('label classification: acc={:.4f}, normal_acc={:.4f}, adv_acc={:.4f}'
            .format(100.0 * cls_acc, 100.0 * cls_acc_normal, 100.0 * cls_acc_adv))

# debug:
# X_test_img     = convert_tensor_to_image(x)
# plt.imshow(X_test_img[img_ind])
# plt.show()
#
# X_test_tta_img = convert_tensor_to_image(inputs.detach().cpu().numpy())
# for t_ind in range(0, 5):
#     plt.imshow(X_test_tta_img[t_ind])
#     plt.show()
"""
This script uses TTA for robustness. No training, only evaluation and using the results to classify normal/adv
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
from active_learning_project.models.projection_head import MLP
from Pointnet_Pointnet2_pytorch.models.pointnet_utils import feature_transform_reguliarzer
from Pointnet_Pointnet2_pytorch.models.pointnet_cls import PointNet

parser = argparse.ArgumentParser(description='PyTorch TTA defense with mlp')
parser.add_argument('--checkpoint_dir',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00',
                    type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='cw_targeted', type=str, help='attack directory')

# eval:
parser.add_argument('--tta_size', default=1000, type=int, help='number of test-time augmentations')
parser.add_argument('--features', default='probs', type=str, help='which features to use from resnet: embeddings/logits/probs')
parser.add_argument('--num_workers', default=10, type=int, help='number of workers')

# transforms:
parser.add_argument('--clip_inputs', action='store_true', help='clipping TTA inputs between 0 and 1')
parser.add_argument('--gaussian_std', default=0.0, type=float, help='Standard deviation of Gaussian noise')  # was 0.0125
parser.add_argument('--soft_transforms', action='store_true', help='applying mellow transforms')

# debug:
parser.add_argument('--test_size', default=200, type=int, help='test size')
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
if args.test_size is not None:
    test_inds = test_inds[0:args.test_size]
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

tta_dataset_test = TTADataset(
    torch.from_numpy(X_test[test_inds]),
    torch.from_numpy(X_test_adv[test_inds]),
    torch.from_numpy(y_test[test_inds]),
    args.tta_size,
    transform=tta_transforms)

test_loader = torch.utils.data.DataLoader(
    tta_dataset_test, batch_size=1, shuffle=False,
    num_workers=args.num_workers, pin_memory=device=='cuda')

all_y_gt        = np.nan * np.ones(2 * test_size)
all_y_is_adv_gt = np.nan * np.ones(2 * test_size)
all_probs       = np.nan * np.ones((2 * test_size, args.tta_size, num_channels), dtype=np.float32)

start_time = time.time()
all_cnt = 0

def rearrange_as_pts(x: torch.Tensor) -> torch.Tensor:
    """Reshape the x tensor from [B * N, D] to [B, N, D]"""
    x = x.reshape(-1, args.tta_size, num_channels)
    return x


with torch.no_grad():
    for batch_idx, (x, y, y_is_adv) in tqdm(enumerate(test_loader)):
        x, y, y_is_adv = x.reshape((-1,) + img_shape), y.reshape(-1), y_is_adv.reshape(-1)
        x, y, y_is_adv = x.to(device), y.to(device), y_is_adv.to(device)
        b = all_cnt
        e = b + y.size(0)
        all_probs[b:e] = rearrange_as_pts(net(x)[args.features]).cpu().numpy()
        all_y_gt[b:e] = y.cpu().numpy()
        all_y_is_adv_gt[b:e] = y_is_adv.cpu().numpy()
        all_cnt += y.size(0)
assert all_cnt == 2 * test_size
assert not np.isnan(all_probs).any()
assert not np.isnan(all_y_gt).any()
assert not np.isnan(all_y_is_adv_gt).any()

all_preds = all_probs.sum(axis=1).argmax(axis=1)
acc = np.mean(all_preds == all_y_gt)
all_normal_indices = np.where(all_y_is_adv_gt == 0)[0]
all_adv_indices = np.where(all_y_is_adv_gt == 1)[0]
norm_acc = np.mean(all_preds[all_normal_indices] == all_y_gt[all_normal_indices])
adv_acc = np.mean(all_preds[all_adv_indices] == all_y_gt[all_adv_indices])

end_time = time.time()
tps = (end_time - start_time) / (2 * test_size)

logger.info('acc={:.4f}, normal_acc={}, adv_acc={}, tps={}'.format(100.0 * acc, 100.0 * norm_acc, 100.0 * adv_acc, tps))

# debug:
# X_test_img     = convert_tensor_to_image(x)
# plt.imshow(X_test_img[img_ind])
# plt.show()
#
# X_test_tta_img = convert_tensor_to_image(inputs.detach().cpu().numpy())
# for t_ind in range(0, 5):
#     plt.imshow(X_test_tta_img[t_ind])
#     plt.show()
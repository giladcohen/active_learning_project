"""
This script uses TTA for robustness. Each TTA is perturbed to maximize the KL divergence between the TTA and
the original example.
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
parser.add_argument('--eval_batch_size', default=100, type=int, help='batch size for the TTA evaluation')

# transforms:
parser.add_argument('--clip_inputs', action='store_true', help='clipping TTA inputs between 0 and 1')
parser.add_argument('--gaussian_std', default=0.0, type=float, help='Standard deviation of Gaussian noise')  # was 0.0125

# training:
parser.add_argument('--train_batch_size', default=6, type=int, help='batch size for the TTA training')
parser.add_argument('--steps', default=2000, type=int, help='training steps for each image')
parser.add_argument('--ema_decay', default=0.998, type=float, help='EMA decay')
parser.add_argument('--val_size', default=200, type=int, help='validation size')
parser.add_argument('--lambda_feat_trans', default=0.001, type=float, help='validation size')

# optimizer:
parser.add_argument('--opt', default='adam', type=str, help='optimizer: sgd, adam, rmsprop, lars')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate on the mlp')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay on the mlp')
parser.add_argument('--mom', default=0.9, type=float, help='momentum of sgd optimizer of beta1 for adam')

# debug:
parser.add_argument('--test_size', default=None, type=int, help='test size')
parser.add_argument('--dump_dir', default='tmp', type=str, help='the dump dir')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

MaxInt32 =  1 << 32 - 1

TRAIN_TIME_CNT = 0.0
TEST_TIME_CNT = 0.0
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

train_adv_det_writer = SummaryWriter(os.path.join(DUMP_DIR, 'train_adv_detection'))

# Data
logger.info('==> Preparing data..')
testloader = get_test_loader(
    dataset=dataset,
    batch_size=args.eval_batch_size,
    num_workers=1,
    pin_memory=True
)

if args.clip_inputs == True:
    clip_min, clip_max = 0.0, 1.0
else:
    clip_min, clip_max = -np.inf, np.inf
p_hflip = 0.5 if 'cifar' in dataset else 0.0
tta_transforms = transforms.Compose([
    my_transforms.ColorJitterPro(
        brightness=[0.6, 1.4],
        contrast=[0.7, 1.3],
        saturation=[0.5, 1.5],
        hue=[-0.06, 0.06],
        gamma=[0.7, 1.3]
    ),
    transforms.Pad(padding=16, padding_mode='edge'),
    transforms.RandomAffine(
        degrees=[-15, 15],
        translate=(4.0 / 64, 4.0 / 64),
        scale=(0.9, 1.1),
        shear=None,
        resample=PIL.Image.BILINEAR,
        fillcolor=None
    ),
    transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.5]),
    transforms.CenterCrop(size=32),
    transforms.RandomHorizontalFlip(p=p_hflip),
    my_transforms.GaussianNoise(0, args.gaussian_std),
    my_transforms.Clip(clip_min, clip_max)
])

global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
classes = testloader.dataset.classes

# get data for TTA loaders
X_test           = get_normalized_tensor(testloader)
y_test           = np.asarray(testloader.dataset.targets)
X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
img_shape = X_test.shape[1:]

# filter by mini_val and mini_test sets
if args.val_size is not None:
    val_inds         = val_inds[0:args.val_size]
if args.test_size is not None:
    test_inds        = test_inds[0:args.test_size]
val_size = len(val_inds)
test_size = len(test_inds)

net = get_model(train_args['net'])(num_classes=len(classes), activation=train_args['activation'])
net = net.to(device)
net.load_state_dict(global_state['best_net'])
net.eval()  # frozen
pointnet = PointNet(k=1, channel=len(classes))
pointnet = pointnet.to(device)

# summary(net, (3, 32, 32))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

def reset_opt():
    global optimizer
    if args.opt == 'sgd':
        optimizer = optim.SGD(
            pointnet.parameters(),
            lr=args.lr,
            momentum=args.mom,
            weight_decay=args.wd,
            nesterov=args.mom > 0)
    elif args.opt == 'adam':
        optimizer = optim.Adam(
            pointnet.parameters(),
            lr=args.lr,
            betas=(args.mom, 0.999),
            weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(
            pointnet.parameters(),
            lr=args.lr,
            betas=(args.mom, 0.999),
            weight_decay=args.wd)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            pointnet.parameters(),
            lr=args.lr,
            momentum=args.mom,
            weight_decay=args.wd)
    else:
        raise AssertionError('optimizer {} is not expected'.format(args.opt))

def flush():
    train_adv_det_writer.flush()


reset_opt()
bce_loss = nn.BCEWithLogitsLoss()

robustness_preds            = -1 * np.ones(test_size, dtype=np.int32)
robustness_preds_adv        = -1 * np.ones(test_size, dtype=np.int32)

# multi TTAs
robustness_probs            = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)
robustness_probs_adv        = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)

tta_dataset = TTADataset(
    torch.from_numpy(X_test[val_inds]),
    torch.from_numpy(X_test_adv[val_inds]),
    args.tta_size,
    transform=tta_transforms)

train_loader = torch.utils.data.DataLoader(
    tta_dataset, batch_size=1, shuffle=True,
    num_workers=4, pin_memory=device=='cuda'
)


def rearrange_as_pts(x: torch.Tensor) -> torch.Tensor:
    """Reshape the x tensor from [B * N, D] to [B, D, N]. y is selected only for B values"""
    x = x.reshape(2, args.tta_size, len(classes))
    x = x.transpose(1, 2)
    # y = [y[i] for i in range(len(y)) if i % args.tta_size == 0]
    # y = torch.as_tensor(y, dtype=torch.float32)
    return x

def train():
    global global_step, epoch, total_loss

    pointnet.train()
    train_loss = 0.0
    predicted = []
    labels = []
    optimizer.zero_grad()

    batch_probs_points = -1 * torch.ones(args.train_batch_size, len(classes), args.tta_size).to(device)
    y = -1 * torch.ones(args.train_batch_size).to(device)

    batch_cnt = 0
    while batch_cnt < args.train_batch_size:
        for batch_idx, (inputs, targets) in enumerate(train_loader):  # train a single step
            b = batch_cnt
            e = b + 2
            inputs = inputs.reshape((-1,) + img_shape)
            targets = targets.reshape(-1)
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                batch_probs_points[b:e] = rearrange_as_pts(net(inputs)['probs'])
            y[b:e] = targets
            batch_cnt += targets.size(0)

            if batch_cnt >= args.train_batch_size:
                break

    assert batch_cnt == args.train_batch_size
    assert (batch_probs_points != -1).all()
    assert (y != -1).all()

    out, trans_feat = pointnet(batch_probs_points)
    out = out.squeeze()
    loss_bce = bce_loss(out, y)
    loss_feat_trans = feature_transform_reguliarzer(trans_feat)
    loss = loss_bce + args.lambda_feat_trans * loss_feat_trans

    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    preds = out.ge(0.0)
    predicted.extend(preds.detach().cpu().numpy())
    labels.extend(y.detach().cpu().numpy())
    num_corrected = preds.eq(y).sum().item()
    acc = num_corrected / y.size(0)

    if global_step % 10 == 0:  # sampling, once ever 10 train iterations
        train_adv_det_writer.add_scalar('losses/loss_bce', loss_bce, global_step)
        train_adv_det_writer.add_scalar('losses/loss_feat_trans', loss_feat_trans, global_step)
        train_adv_det_writer.add_scalar('metrics/acc', 100.0 * acc, global_step)

    global_step += 1

    predicted = np.asarray(predicted)
    labels = np.asarray(labels)
    train_acc = 100.0 * np.mean(predicted == labels)
    logger.info('Epoch #{} (TRAIN): loss={}\tacc={:.2f}'.format(epoch + 1, train_loss, train_acc))

global_step = 0
logger.info('Testing randomized net...')
# test()
logger.info('start training {} steps...'.format(args.steps))
for epoch in tqdm(range(args.steps)):
    train()
    # if epoch % 100:
    #     test()
# test()
flush()
exit(0)

# debug:
X_test_img     = convert_tensor_to_image(x)
plt.imshow(X_test_img[img_ind])
plt.show()

X_test_tta_img = convert_tensor_to_image(inputs.detach().cpu().numpy())
for t_ind in range(0, 5):
    plt.imshow(X_test_tta_img[t_ind])
    plt.show()
"""
This script uses TTA for robustness and adv classification.
We pass the network's features into a pointnet which classifies the truth label, and whether or not the image is adv.
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
from active_learning_project.datasets.tta_pair_dataset import TTAPairDataset
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor, \
    get_single_img_dataloader, get_explicit_train_loader
from active_learning_project.utils import EMA, update_moving_average, convert_tensor_to_image, set_logger, get_model, \
    reset_net, pytorch_evaluate
from active_learning_project.datasets.utils import get_dataset_inds
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
parser.add_argument('--tta_size', default=100, type=int, help='number of test-time augmentations')

# transforms:
parser.add_argument('--clip_inputs', action='store_true', help='clipping TTA inputs between 0 and 1')
parser.add_argument('--gaussian_std', default=0.005, type=float, help='Standard deviation of Gaussian noise')  # was 0.0125
parser.add_argument('--soft_transforms', action='store_true', help='applying mellow transforms')

# training:
parser.add_argument('--train_batch_size', default=128, type=int, help='batch size for the TTA training')
parser.add_argument('--steps', default=2000, type=int, help='training steps for each image')
parser.add_argument('--ema_decay', default=0.998, type=float, help='EMA decay')
parser.add_argument('--val_size', default=None, type=int, help='validation size')
parser.add_argument('--lambda_feat_trans', default=0.0, type=float, help='regularization on ')
parser.add_argument('--lambda_adv_det', default=1.0, type=float, help='adversarial detection coefficient')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')

# architecture
parser.add_argument('--features', default='probs', type=str, help='which features to use from resnet: embeddings/logits/probs')

# optimizer:
parser.add_argument('--opt', default='adam', type=str, help='optimizer: sgd, adam, rmsprop, lars')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate on the pointnet')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay on the pointnet')
parser.add_argument('--mom', default=0.9, type=float, help='momentum of sgd optimizer of beta1 for adam')

# debug:
parser.add_argument('--test_size', default=500, type=int, help='test size')
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
val_inds, test_inds = get_dataset_inds(dataset)

with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

rand_gen = np.random.RandomState(12345)

train_defense_writer = SummaryWriter(os.path.join(DUMP_DIR, 'train_adv_detection'))
test_defense_writer = SummaryWriter(os.path.join(DUMP_DIR, 'test_adv_detection'))

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
    test_inds        = test_inds[0:args.test_size]
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

pointnet = PointNet(k=len(classes) + 1, channel=num_channels)
pointnet = pointnet.to(device)

# summary(net, (3, 32, 32))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

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
    train_defense_writer.flush()


bce_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss()

tta_dataset_train = TTAPairDataset(
    torch.from_numpy(X_test[val_inds]),
    torch.from_numpy(X_test_adv[val_inds]),
    torch.from_numpy(y_test[val_inds]),
    args.tta_size,
    transform=tta_transforms)

tta_dataset_test = TTAPairDataset(
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
    """Reshape the x tensor from [B * N, D] to [B, D, N]"""
    x = x.reshape(-1, args.tta_size, num_channels)
    x = x.transpose(1, 2)
    return x

def train():
    global global_step
    net.eval()
    pointnet.train()
    optimizer.zero_grad()
    start_time = time.time()

    net_features = np.nan * torch.ones(args.train_batch_size, num_channels, args.tta_size).to(device)  # (B, D, N)
    y_gt = np.nan * torch.ones(args.train_batch_size).to(device)  # B
    y_is_adv_gt = np.nan * torch.ones(args.train_batch_size).to(device)  # B

    batch_cnt = 0
    while batch_cnt < args.train_batch_size:
        for batch_idx, (x, yb, yb_is_adv) in enumerate(train_loader):
            x, yb, yb_is_adv = x.reshape((-1,) + img_shape), yb.reshape(-1), yb_is_adv.reshape(-1)
            x, yb, yb_is_adv = x.to(device), yb.to(device), yb_is_adv.to(device)
            b = batch_cnt
            e = b + yb.size(0)
            assert not x.isnan().any()
            with torch.no_grad():
                net_features[b:e] = rearrange_as_pts(net(x)[args.features])
            y_gt[b:e] = yb
            y_is_adv_gt[b:e] = yb_is_adv
            batch_cnt += yb.size(0)

            if batch_cnt >= args.train_batch_size:
                break

    assert batch_cnt == args.train_batch_size
    assert not net_features.isnan().any()
    assert not y_gt.isnan().any()
    assert not y_is_adv_gt.isnan().any()

    out, trans_feat = pointnet(net_features)
    y_probs = out[:, :len(classes)]
    y_is_adv_probs = out[:, -1]
    loss_ce = ce_loss(y_probs, y_gt.long())
    loss_bce = bce_loss(y_is_adv_probs, y_is_adv_gt)
    loss_feat_trans = feature_transform_reguliarzer(trans_feat)
    loss = loss_ce + args.lambda_adv_det * loss_bce + args.lambda_feat_trans * loss_feat_trans

    loss.backward()
    optimizer.step()

    y_preds = y_probs.argmax(dim=1)
    y_is_adv_preds = y_is_adv_probs.ge(0.0)
    num_corrected_classification = y_preds.eq(y_gt).sum().item()
    num_corrected_is_adv = y_is_adv_preds.eq(y_is_adv_gt).sum().item()
    classification_acc = num_corrected_classification / y_gt.size(0)
    is_adv_acc = num_corrected_is_adv / y_gt.size(0)
    end_time = time.time()

    if global_step % 1 == 0:  # sampling, once ever 1 train iterations
        train_defense_writer.add_scalar('losses/loss_ce', loss_bce, global_step)
        train_defense_writer.add_scalar('losses/loss_bce', loss_bce, global_step)
        train_defense_writer.add_scalar('losses/loss_feat_trans', loss_feat_trans, global_step)
        train_defense_writer.add_scalar('losses/loss', loss, global_step)
        train_defense_writer.add_scalar('metrics/classification_acc', 100.0 * classification_acc, global_step)
        train_defense_writer.add_scalar('metrics/is_adv_acc', 100.0 * is_adv_acc, global_step)
        train_defense_writer.add_scalar('stats/secs_per_image', (end_time - start_time) / args.train_batch_size, global_step)

    logger.info('global step #{} (TRAIN): loss={}\tclassification acc={:.4f}, is adv acc={:.4f}'
                .format(global_step + 1, loss, 100.0 * classification_acc, 100.0 * is_adv_acc))

def test():
    start_time = time.time()
    net.eval()
    pointnet.eval()
    y_preds = np.nan * np.ones(2 * test_size)
    y_is_adv_preds = np.nan * np.ones(2 * test_size)
    y_gt = np.nan * np.ones(2 * test_size)
    y_is_adv_gt = np.nan * np.ones(2 * test_size)

    all_cnt = 0
    loss = 0.0
    loss_ce = 0.0
    loss_bce = 0.0
    loss_feat_trans = 0.0

    for batch_idx, (x, yb, yb_is_adv) in enumerate(test_loader):
        x, yb, yb_is_adv = x.reshape((-1,) + img_shape), yb.reshape(-1), yb_is_adv.reshape(-1)
        x, yb, yb_is_adv = x.to(device), yb.to(device), yb_is_adv.to(device)
        b = all_cnt
        e = b + yb.size(0)
        assert not x.isnan().any()
        net_features = rearrange_as_pts(net(x)[args.features])
        out, trans_feat = pointnet(net_features)
        y_probs = out[:, :len(classes)]
        y_is_adv_probs = out[:, -1]

        loss_ce += ce_loss(y_probs, yb)
        loss_bce += bce_loss(y_is_adv_probs, yb_is_adv)
        loss_feat_trans += feature_transform_reguliarzer(trans_feat)
        loss = loss_ce + args.lambda_adv_det * loss_bce + args.lambda_feat_trans * loss_feat_trans

        y_preds[b:e] = y_probs.argmax(dim=1).cpu().numpy()
        y_is_adv_preds[b:e] = y_is_adv_probs.ge(0.0).cpu().numpy()
        y_gt[b:e] = yb.cpu().numpy()
        y_is_adv_gt[b:e] = yb_is_adv.cpu().numpy()

        all_cnt += yb.size(0)

    assert all_cnt == 2 * test_size
    assert not (np.isnan(y_preds)).any()
    assert not (np.isnan(y_is_adv_preds)).any()
    assert not (np.isnan(y_gt)).any()
    assert not (np.isnan(y_is_adv_gt)).any()

    # averaging losses
    loss /= (2 * test_size)
    loss_ce /= (2 * test_size)
    loss_feat_trans /= (2 * test_size)
    loss_bce /= (2 * test_size)

    classification_acc = np.mean(y_preds == y_gt)
    is_adv_acc = np.mean(y_is_adv_preds == y_is_adv_gt)

    all_normal_indices = np.where(y_is_adv_gt == 0)[0]
    all_adv_indices = np.where(y_is_adv_gt == 1)[0]
    norm_classification_acc = np.mean(y_preds[all_normal_indices] == y_gt[all_normal_indices])
    adv_classification_acc = np.mean(y_preds[all_adv_indices] == y_gt[all_adv_indices])
    norm_is_adv_acc = np.mean(y_is_adv_preds[all_normal_indices] == y_is_adv_gt[all_normal_indices])
    adv_is_adv_acc = np.mean(y_is_adv_preds[all_adv_indices] == y_is_adv_gt[all_adv_indices])

    end_time = time.time()

    test_defense_writer.add_scalar('losses/loss_ce', loss_ce, global_step)
    test_defense_writer.add_scalar('losses/loss_bce', loss_bce, global_step)
    test_defense_writer.add_scalar('losses/loss_feat_trans', loss_feat_trans, global_step)
    test_defense_writer.add_scalar('losses/loss', loss, global_step)
    test_defense_writer.add_scalar('metrics/classification_acc', 100.0 * classification_acc, global_step)
    test_defense_writer.add_scalar('metrics/is_adv_acc', 100.0 * is_adv_acc, global_step)
    test_defense_writer.add_scalar('metrics/norm_classification_acc', 100.0 * norm_classification_acc, global_step)
    test_defense_writer.add_scalar('metrics/adv_classification_acc', 100.0 * adv_classification_acc, global_step)
    test_defense_writer.add_scalar('metrics/norm_is_adv_acc', 100.0 * norm_is_adv_acc, global_step)
    test_defense_writer.add_scalar('metrics/adv_is_adv_acc', 100.0 * adv_is_adv_acc, global_step)
    test_defense_writer.add_scalar('stats/secs_per_image', (end_time - start_time) / test_size, global_step)


    logger.info('global step #{} (TEST): loss={}\tclassification acc={:.4f}, is adv acc={:.4f}'
                .format(global_step + 1, loss, 100.0 * classification_acc, 100.0 * is_adv_acc))

global_step = 0
logger.info('Testing randomized net...')
with torch.no_grad():
    test()
logger.info('start training {} steps...'.format(args.steps))
for global_step in tqdm(range(args.steps)):
    train()
    if global_step % 20 == 0 and global_step > 0:
        with torch.no_grad():
            test()
with torch.no_grad():
    test()

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
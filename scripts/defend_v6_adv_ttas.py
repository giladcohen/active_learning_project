"""
This script uses TTA for robustness. Each TTA is perturbed to maximize the KL divergence between the TTA and
the original example.
"""
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
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor, \
    get_single_img_dataloader
from active_learning_project.utils import EMA, update_moving_average, convert_tensor_to_image, set_logger, get_model
from active_learning_project.metric_utils import calc_first_n_adv_acc, calc_first_n_adv_acc_from_probs_summation
from active_learning_project.losses.losses import VAT

parser = argparse.ArgumentParser(description='PyTorch TTA defense V4')
parser.add_argument('--checkpoint_dir',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00',
                    type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='cw_targeted', type=str, help='attack directory')

# eval
parser.add_argument('--tta_size', default=1000, type=int, help='number of test-time augmentations')
parser.add_argument('--eval_batch_size', default=100, type=int, help='batch size for the TTA evaluation')
parser.add_argument('--mini_test', action='store_true', help='test only 2500 mini_test_inds')

# transforms:
parser.add_argument('--clip_inputs', action='store_true', help='clipping TTA inputs between 0 and 1')
parser.add_argument('--gaussian_std', default=0.0, type=float, help='Standard deviation of Gaussian noise')  # was 0.0125
parser.add_argument('--n_power', default=1, type=int, help='VAT number of adversarial steps')
parser.add_argument('--xi', default=1e-6, type=float, help='VAT factor to multiply the adv perturbation noise')
parser.add_argument('--radius', default=3.5, type=float, help='VAT perturbation 2-norm ball radius')

# debug:
parser.add_argument('--debug_size', default=None, type=int, help='number of image to run in debug mode')
parser.add_argument('--dump_dir', default='tmp', type=str, help='the dump dir')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()
args.mini_test = True

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
args.dataset = train_args['dataset']

with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

rand_gen = np.random.RandomState(12345)
eval_normal_writer = SummaryWriter(os.path.join(DUMP_DIR, 'eval_normal'))
eval_adv_writer    = SummaryWriter(os.path.join(DUMP_DIR, 'eval_adv'))

# Data
logger.info('==> Preparing data..')
testloader = get_test_loader(
    dataset=train_args['dataset'],
    batch_size=args.eval_batch_size,
    num_workers=1,
    pin_memory=True
)

if args.clip_inputs == True:
    clip_min, clip_max = 0.0, 1.0
else:
    clip_min, clip_max = -np.inf, np.inf

p_hflip = 0.5 if 'cifar' in args.dataset else 0.0
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
X_test           = get_normalized_tensor(testloader)
y_test           = np.asarray(testloader.dataset.targets)
X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
if targeted:
    y_test_adv   = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))
y_test_preds     = np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy'))
y_test_adv_preds = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))
test_size = len(X_test)

# summary(net, (3, 32, 32))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

def reset_net():
    global net
    net = get_model(train_args['net'])(num_classes=len(classes), activation=train_args['activation'])
    net = net.to(device)
    net.load_state_dict(global_state['best_net'])
    net.eval()
    # if 'vat' in globals():
    #     vat.model = net
    try:
        vat.model = net
    except Exception as e:
        logger.info('VAT.model = net did not succeed because:\n {}'.format(e))

reset_net()
vat = VAT(net, args.n_power, args.xi, args.radius)

# test images inds:
mini_test_inds = np.load(os.path.join(args.checkpoint_dir, 'mini_test_inds.npy'))
if args.mini_test:
    all_test_inds = mini_test_inds
else:
    all_test_inds = np.arange(len(X_test))
if args.debug_size is not None:
    all_test_inds = all_test_inds[:args.debug_size]
img_cnt = len(all_test_inds)

robustness_preds            = -1 * np.ones(test_size, dtype=np.int32)
robustness_preds_adv        = -1 * np.ones(test_size, dtype=np.int32)

# multi TTAs
robustness_probs            = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)
robustness_probs_adv        = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)

def eval(set):
    global TEST_TIME_CNT, net
    if set == 'normal':
        x = X_test
        rob_preds     = robustness_preds
        rob_probs     = robustness_probs
    else:
        x = X_test_adv
        rob_preds     = robustness_preds_adv
        rob_probs     = robustness_probs_adv

    start_time = time.time()
    net.eval()

    orig_logits = net(torch.from_numpy(np.expand_dims(x[img_ind], 0)).to(device))['logits']
    rob_preds[img_ind] = orig_logits.argmax(dim=1).squeeze().detach().cpu().numpy()
    tta_cnt = 0
    for batch_idx, (inputs, targets) in enumerate(tta_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        b = tta_cnt
        e = min(tta_cnt + len(inputs), args.tta_size)
        orig_tta_logits = net(inputs)['logits']
        pert_inputs = vat.virtual_adversarial_images(inputs[0:(e-b)], orig_tta_logits)
        rob_probs[img_ind, b:e] = net(pert_inputs)['probs'][0:(e-b)].detach().cpu().numpy()
        tta_cnt += e-b
        if tta_cnt >= args.tta_size:
            break
    assert tta_cnt == args.tta_size, 'tta_cnt={} must match the args.tta_size'.format(tta_cnt)
    TEST_TIME_CNT += time.time() - start_time


for i in tqdm(range(img_cnt)):
    # for i in range(img_cnt):  # debug
    img_ind = all_test_inds[i]
    # normal
    tta_loader = get_single_img_dataloader(args.dataset, X_test, y_test, args.eval_batch_size, args.tta_size,
                                           pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    eval('normal')

    # adv
    tta_loader = get_single_img_dataloader(args.dataset, X_test_adv, y_test, args.eval_batch_size, args.tta_size,
                                           pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    eval('adv')

    acc_all, acc_all_adv = calc_first_n_adv_acc(robustness_preds, robustness_preds_adv, y_test, all_test_inds, i + 1)
    tta_acc_all, tta_acc_all_adv = calc_first_n_adv_acc_from_probs_summation(robustness_probs, robustness_probs_adv, y_test, all_test_inds, i + 1)

    logger.info('accuracy on the fly after {} samples: original image: {:.2f}/{:.2f}%, TTAs: {:.2f}/{:.2f}%'
        .format(i + 1, acc_all * 100, acc_all_adv * 100, tta_acc_all * 100, tta_acc_all_adv * 100))

average_test_time = TEST_TIME_CNT / (2 * img_cnt)
logger.info('average eval time per sample: {} secs'.format(average_test_time))

logger.info('done')
logging.shutdown()
exit(0)


# debug:
X_test_img     = convert_tensor_to_image(x)
plt.imshow(X_test_img[img_ind])
plt.show()

X_test_tta_img = convert_tensor_to_image(inputs.detach().cpu().numpy())
for t_ind in range(0, 5):
    plt.imshow(X_test_tta_img[t_ind])
    plt.show()
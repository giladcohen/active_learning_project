"""
This script uses TTA for robustness
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

from utils import convert_image_to_tensor
import matplotlib.pyplot as plt

sys.path.insert(0, "..")

from active_learning_project.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101
import active_learning_project.datasets.my_transforms as my_transforms
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor, \
    get_single_img_dataloader
from active_learning_project.utils import EMA, update_moving_average, convert_tensor_to_image

parser = argparse.ArgumentParser(description='PyTorch CLR training on base pretrained net')
parser.add_argument('--checkpoint_dir',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00',
                    type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='cw_targeted', type=str, help='attack directory')

# eval
parser.add_argument('--tta_size', default=1000, type=int, help='number of test-time augmentations')
parser.add_argument('--batch_size', default=100, type=int, help='batch size for the TTA evaluation')

# debug:
parser.add_argument('--debug_size', default=None, type=int, help='number of image to run in debug mode')
parser.add_argument('--dump_dir', default='tmp', type=str, help='the dump dir')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

TEST_TIME_CNT = 0.0
ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
DUMP_DIR = os.path.join(ATTACK_DIR, args.dump_dir)
os.makedirs(DUMP_DIR, exist_ok=True)

# dumping args to txt file
with open(os.path.join(DUMP_DIR, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

logging.basicConfig(filename=os.path.join(DUMP_DIR, 'log.log'),
                    filemode='w',
                    format='%(asctime)s %(name)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

# logger = logging.getLogger()
def log(str):
    logging.info(str)
    print(str)

def calc_robust_metrics(robustness_preds, robustness_preds_adv):
    acc_all = np.mean(robustness_preds[all_test_inds] == y_test[all_test_inds])
    acc_all_adv = np.mean(robustness_preds_adv[all_test_inds] == y_test[all_test_inds])
    return acc_all, acc_all_adv
    # log('Robust classification accuracy: all samples: {:.2f}/{:.2f}%'.format(acc_all * 100, acc_all_adv * 100))

def calc_first_n_robust_metrics(robustness_preds, robustness_preds_adv, n):
    acc_all = np.mean(robustness_preds[all_test_inds][0:n] == y_test[all_test_inds][0:n])
    acc_all_adv = np.mean(robustness_preds_adv[all_test_inds][0:n] == y_test[all_test_inds][0:n])
    return acc_all, acc_all_adv

def calc_first_n_robust_metrics_from_probs_summation(tta_robustness_probs, tta_robustness_probs_adv, n):
    tta_robustness_probs_sum = tta_robustness_probs.sum(axis=1)
    robustness_preds = tta_robustness_probs_sum.argmax(axis=1)
    tta_robustness_probs_adv_sum = tta_robustness_probs_adv.sum(axis=1)
    robustness_preds_adv = tta_robustness_probs_adv_sum.argmax(axis=1)
    acc_all = np.mean(robustness_preds[all_test_inds][0:n] == y_test[all_test_inds][0:n])
    acc_all_adv = np.mean(robustness_preds_adv[all_test_inds][0:n] == y_test[all_test_inds][0:n])
    return acc_all, acc_all_adv

def get_logits_from_emb_center(tta_embedding, model):
    tta_embeddings_center = tta_embedding.mean(axis=0)
    logits = model.linear(tta_embeddings_center)
    return logits

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
args.dataset = train_args['dataset']

with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

rand_gen = np.random.RandomState(12345)
normal_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'normal_debug'))
adv_writer    = SummaryWriter(os.path.join(args.checkpoint_dir, 'adv_debug'))
batch_size = args.batch_size

# Data
log('==> Preparing data..')
testloader = get_test_loader(
    dataset=train_args['dataset'],
    batch_size=100,
    num_workers=1,
    pin_memory=True
)

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
    my_transforms.GaussianNoise(0, 0.015),
    my_transforms.Clip(0.0, 1.0)
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

# Model
if train_args['net'] == 'resnet18':
    model = ResNet18
elif train_args['net'] == 'resnet34':
    model = ResNet34
elif train_args['net'] == 'resnet50':
    model = ResNet50
elif train_args['net'] == 'resnet101':
    model = ResNet101
else:
    raise AssertionError("network {} is unknown".format(train_args['net']))

def reset_net():
    global net
    net = model(num_classes=len(classes), activation=train_args['activation'])
    net = net.to(device)
    net.load_state_dict(global_state['best_net'])
    net.eval()

reset_net()

# test images inds:
all_test_inds = np.arange(len(X_test))
if args.debug_size is not None:
    all_test_inds = all_test_inds[:args.debug_size]
img_cnt = len(all_test_inds)

robustness_preds            = -1 * np.ones(test_size, dtype=np.int32)
robustness_preds_adv        = -1 * np.ones(test_size, dtype=np.int32)

# multi TTAs
robustness_probs            = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)
robustness_probs_adv        = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)

def kl_loss(s_logits, t_logits):
    ret1 = F.kl_div(F.log_softmax(s_logits, dim=1), F.softmax(t_logits, dim=1), reduction="batchmean")
    ret2 = F.kl_div(F.log_softmax(t_logits, dim=1), F.softmax(s_logits, dim=1), reduction="batchmean")
    return 0.5 * (ret1 + ret2)


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

    with torch.no_grad():
        rob_preds[img_ind] = net(torch.from_numpy(np.expand_dims(x[img_ind], 0)).to(device))['preds'].squeeze().detach().cpu().numpy()
        tta_cnt = 0
        while tta_cnt < args.tta_size:
            inputs, targets = list(tta_loader)[0]
            inputs, targets = inputs.to(device), targets.to(device)
            b = tta_cnt
            e = min(tta_cnt + len(inputs), args.tta_size)
            rob_probs[img_ind, b:e] = net(inputs)['probs'][0:(e-b)].detach().cpu().numpy()
            tta_cnt += e-b
            assert tta_cnt <= args.tta_size, 'not cool!'

    TEST_TIME_CNT += time.time() - start_time

for i in tqdm(range(img_cnt)):
    # for i in range(img_cnt):  # debug
    img_ind = all_test_inds[i]
    # normal
    tta_loader = get_single_img_dataloader(args.dataset, X_test, y_test, args.batch_size,
                                           pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    eval('normal')

    # adv
    tta_loader = get_single_img_dataloader(args.dataset, X_test_adv, y_test, args.batch_size,
                                           pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    eval('adv')

    acc_all, acc_all_adv = calc_first_n_robust_metrics(robustness_preds, robustness_preds_adv, i + 1)
    tta_acc_all, tta_acc_all_adv = calc_first_n_robust_metrics_from_probs_summation(robustness_probs, robustness_probs_adv, i + 1)

    log('accuracy on the fly after {} samples: original image: {:.2f}/{:.2f}%, TTAs: {:.2f}/{:.2f}%'
        .format(i + 1, acc_all * 100, acc_all_adv * 100, tta_acc_all * 100, tta_acc_all_adv * 100))

average_test_time = TEST_TIME_CNT / (2 * img_cnt)
log('average eval time per sample: {} secs'.format(average_test_time))

log('done')
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
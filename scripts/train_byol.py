"""
This script loads a pretrained checkpoint and used the base model to extract features (h), which on top of them we
calculate the contrastive loss using a simple MLP g(h).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn
import logging
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import time
import sys
import PIL
import scipy
import copy
import pickle
from datetime import datetime

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")
sys.path.insert(0, "./byol_pytorch")

from byol_pytorch import BYOL

from art.estimators.classification import PyTorchClassifier

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.models.projection_head import ProjectobHead
import active_learning_project.datasets.my_transforms as my_transforms
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_all_data_loader, get_normalized_tensor, get_single_img_dataloader
from active_learning_project.utils import remove_substr_from_keys, boolean_string, save_features, pytorch_evaluate, \
    majority_vote
from torchsummary import summary
import torchvision.transforms as transforms

eps = 1e-8

parser = argparse.ArgumentParser(description='PyTorch CLR training on base pretrained net')
parser.add_argument('--checkpoint_dir',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00',
                    type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='cw_targeted', type=str, help='attack directory')

# train
parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
parser.add_argument('--steps', default=50, type=int, help='number of training steps')
parser.add_argument('--batch_size', default=32, type=int, help='batch size for the CLR training')
parser.add_argument('--opt', default='adam', type=str, help='optimizer: sgd, adam, rmsprop')
parser.add_argument('--mom', default=0.0, type=float, help='momentum of optimizer')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
parser.add_argument('--lambda_cont', default=1.0, type=float, help='weight of similarity loss')
parser.add_argument('--lambda_ent', default=0.0, type=float, help='Regularization for entropy loss')
parser.add_argument('--lambda_wdiff', default=0.0, type=float, help='Regularization for weight diff')

# eval
parser.add_argument('--tta_size', default=50, type=int, help='number of test-time augmentations in eval phase')
parser.add_argument('--mini_test', action='store_true', help='test only 1000 mini_test_inds')

# debug:
parser.add_argument('--debug_size', default=100, type=int, help='number of image to run in debug mode')
parser.add_argument('--dump', '-d', action='store_true', help='get debug stats')
parser.add_argument('--dump_dir', default='dump', type=str, help='the dump dir')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

args.mini_test = True  # always use mini testset
TRAIN_TIME_CNT = 0.0
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
    global net, learner

    try:
        del net
    except:
        print('net was not constructed yet')

    try:
        del learner
    except:
        print('learner was not constructed yet')

    # Model
    if train_args['net'] == 'resnet34':
        net      = ResNet34(num_classes=len(classes), activation=train_args['activation'])
    elif train_args['net'] == 'resnet101':
        net      = ResNet101(num_classes=len(classes), activation=train_args['activation'])
    else:
        raise AssertionError("network {} is unknown".format(train_args['net']))
    net = net.to(device)

    net.load_state_dict(global_state['best_net'])
    learner = BYOL(net=net, image_size=32, hidden_layer='avgpool', augment_fn=tta_transforms, moving_average_decay=0.9)

def reset_opt():
    global optimizer
    if args.opt == 'sgd':
        optimizer = optim.SGD(
            learner.parameters(),
            lr=args.lr,
            momentum=args.mom,
            weight_decay=args.wd,
            nesterov=args.mom > 0)
    elif args.opt == 'adam':
        optimizer = optim.Adam(
            learner.parameters(),
            lr=args.lr,
            weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(
            learner.parameters(),
            lr=args.lr,
            weight_decay=args.wd)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            learner.parameters(),
            lr=args.lr,
            momentum=args.mom,
            weight_decay=args.wd)
    else:
        raise AssertionError('optimizer {} is not expected'.format(args.opt))

# test images inds:
mini_test_inds = np.load(os.path.join(ATTACK_DIR, 'inds', 'mini_test_inds.npy'))
if args.mini_test:
    all_test_inds = mini_test_inds
else:
    all_test_inds = np.arange(len(X_test))
if args.debug_size is not None:
    all_test_inds = all_test_inds[:args.debug_size]
img_cnt = len(all_test_inds)

robustness_preds            = -1 * np.ones(test_size, dtype=np.int32)
robustness_preds_adv        = -1 * np.ones(test_size, dtype=np.int32)
robustness_probs            = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)
robustness_probs_adv        = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)

def train(set):
    """set='normal' or 'adv'"""
    global TRAIN_TIME_CNT, loss_cont, loss_ent, loss_weight_diff

    start_time = time.time()
    reset_net()
    reset_opt()

    for step in range(args.steps):
        (inputs, targets) = list(train_loader)[0]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = learner(inputs)
        loss.backward()
        optimizer.step()
        learner.update_moving_average()

    TRAIN_TIME_CNT += time.time() - start_time

def test(set):
    global TEST_TIME_CNT
    classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=None,
                                   optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(classes))
    if set == 'normal':
        x = X_test
        rob_preds     = robustness_preds
        rob_probs     = robustness_probs
    else:
        x = X_test_adv
        rob_preds     = robustness_preds_adv
        rob_probs     = robustness_probs_adv

    start_time = time.time()
    rob_preds[img_ind] = classifier.predict(np.expand_dims(x[img_ind], 0)).squeeze().argmax()
    with torch.no_grad():
        tta_cnt = 0
        while tta_cnt < args.tta_size:
            (inputs, targets) = list(train_loader)[0]
            inputs, targets = inputs.to(device), targets.to(device)
            b = tta_cnt
            e = min(tta_cnt + len(inputs), args.tta_size)
            out = net(inputs)
            rob_probs[img_ind, b:e] = out['probs'][0:(e-b)].detach().cpu().numpy()
            tta_cnt += e-b
            assert tta_cnt <= args.tta_size, 'not cool!'

    TEST_TIME_CNT += time.time() - start_time

for i in tqdm(range(img_cnt)):
    # for i in range(img_cnt):  # debug
    img_ind = all_test_inds[i]
    # normal
    train_loader = get_single_img_dataloader(args.dataset, X_test, y_test, 2 * args.batch_size,
                                             pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    train('normal')
    test('normal')

    # adv
    train_loader = get_single_img_dataloader(args.dataset, X_test_adv, y_test, 2 * args.batch_size,
                                             pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    train('adv')
    test('adv')

    acc_all, acc_all_adv = calc_first_n_robust_metrics(robustness_preds, robustness_preds_adv, i + 1)
    tta_acc_all, tta_acc_all_adv = calc_first_n_robust_metrics_from_probs_summation(robustness_probs, robustness_probs_adv, i)
    print('accuracy on the fly (so far) after {} samples: original image: {:.2f}/{:.2f}%, TTAs: {:.2f}/{:.2f}%'
          .format(i, acc_all * 100, acc_all_adv * 100, tta_acc_all * 100, tta_acc_all_adv * 100))

print('done')
exit(0)
# debug:
import matplotlib.pyplot as plt
from active_learning_project.utils import convert_tensor_to_image
# X_test_img       = convert_tensor_to_image(x.detach().cpu().numpy())
X_test_img       = convert_tensor_to_image(X_test)
X_tta_test_img_1 = convert_tensor_to_image(image_one.detach().cpu().numpy())
X_tta_test_img_2 = convert_tensor_to_image(image_two.detach().cpu().numpy())

ind = 9
plt.imshow(X_test_img[1])
plt.show()
plt.imshow(X_tta_test_img_1[ind])
plt.show()
plt.imshow(X_tta_test_img_2[ind])
plt.show()
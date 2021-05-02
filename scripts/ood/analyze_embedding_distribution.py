"""
In this script I analyze the embedding distribution for the augmentations of CIFAR-100 on a network trained on CIFAR-10
"""
from typing import Tuple
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
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors
from numba import njit
import pickle

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.models.resnet import ResNet34, ResNet50, ResNet101
import active_learning_project.datasets.my_transforms as my_transforms
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor, \
    get_single_img_dataloader, get_loader_with_specific_inds
from active_learning_project.utils import convert_tensor_to_image, pytorch_evaluate

from torchsummary import summary
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch CLR training on base pretrained net')
parser.add_argument('--checkpoint_dir',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00',
                    type=str, help='checkpoint dir')

# ood set
parser.add_argument('--ood_set', default='cifar100', type=str, help='OOD set: cifar10, cifar100, or svhn')

# eval
parser.add_argument('--tta_size', default=50, type=int, help='number of test-time augmentations in eval phase')

# debug:
parser.add_argument('--dump_dir', default='dump_only_200', type=str, help='the dump dir')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

TEST_TIME_CNT = 0.0
OOD_DIR = os.path.join(args.checkpoint_dir, 'ood', args.ood_set)
DUMP_DIR = os.path.join(OOD_DIR, args.dump_dir)
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

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
args.dataset = train_args['dataset']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

rand_gen = np.random.RandomState(12345)

# Data
log('==> Preparing data..')
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
train_inds = np.asarray(global_state['train_inds'])
val_inds = np.asarray(global_state['val_inds'])

trainvalloader = get_loader_with_specific_inds(
    dataset=train_args['dataset'],
    batch_size=100,
    is_training=False,
    indices=train_inds,
    num_workers=1,
    pin_memory=True
)

testloader = get_test_loader(
    dataset=train_args['dataset'],
    batch_size=100,
    num_workers=1,
    pin_memory=True
)

ood_testloader = get_test_loader(
    dataset=args.ood_set,
    batch_size=100,
    num_workers=1,
    pin_memory=True
)

# setting up original dataset transforms
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

# setting up OOD dataset transforms
p_hflip = 0.5 if 'cifar' in args.ood_set else 0.0
ood_tta_transforms = transforms.Compose([
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

classes = testloader.dataset.classes
ood_classes = ood_testloader.dataset.classes
X_train          = get_normalized_tensor(trainvalloader)
y_train          = np.asarray(trainvalloader.dataset.targets)
X_test           = get_normalized_tensor(testloader)
y_test           = np.asarray(testloader.dataset.targets)
ood_X_test       = get_normalized_tensor(ood_testloader)
ood_y_test       = np.asarray(ood_testloader.dataset.targets)

test_size  = len(testloader.dataset)
ood_test_size = len(ood_testloader.dataset)
assert test_size == ood_test_size
test_inds  = np.arange(test_size)

# Model
print('==> Building model..')
if train_args['net'] == 'resnet34':
    net = ResNet34(num_classes=len(classes), activation=train_args['activation'])
elif train_args['net'] == 'resnet50':
    net = ResNet50(num_classes=len(classes), activation=train_args['activation'])
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

# setting NN model
NN_L1 = NearestNeighbors(n_neighbors=1, p=1, n_jobs=20)
NN_L2 = NearestNeighbors(n_neighbors=1, p=2, n_jobs=20)
x_train_emb = pytorch_evaluate(net, trainvalloader, ['embeddings'])[0]
NN_L1.fit(x_train_emb)
NN_L2.fit(x_train_emb)

# metrics
metric_keys = ['avg_l2_dist', 'avg_l1_dist', 'max_l2_dist', 'max_l1_dist', 'entropy',
               'avg_l2_dist_to_nn', 'avg_l1_dist_to_nn', 'min_l2_dist_to_nn', 'min_l1_dist_to_nn']
metrics = {}
for key in metric_keys:
    metrics[key] = -1 * np.ones((200, 2))

# metrics functions
@njit
def avg_l1_dist(x: np.ndarray) -> float:
    sum = 0.0
    lenn = len(x)
    n = lenn * (lenn - 1) / 2
    for i in range(0, lenn):
        for j in range(i + 1, lenn):
            sum += np.linalg.norm(x[i] - x[j], ord=1)
    return sum / n

@njit
def avg_l2_dist(x: np.ndarray) -> float:
    sum = 0.0
    lenn = len(x)
    n = lenn * (lenn - 1) / 2
    for i in range(0, lenn):
        for j in range(i + 1, lenn):
            sum += np.linalg.norm(x[i] - x[j], ord=2)
    return sum / n

@njit
def max_l2_dist(x: np.ndarray) -> float:
    maxx = -np.inf
    lenn = len(x)
    for i in range(0, lenn):
        for j in range(i + 1, lenn):
            dist = np.linalg.norm(x[i] - x[j], ord=2)
            if dist > maxx:
                maxx = dist
    return dist

@njit
def max_l1_dist(x: np.ndarray) -> float:
    maxx = -np.inf
    for i in range(0, len(x)):
        for j in range(i + 1, len(x)):
            dist = np.linalg.norm(x[i] - x[j], ord=1)
            if dist > maxx:
                maxx = dist
    return dist

@njit
def entropy(x: np.ndarray) -> float:
    ent = x * np.log(x)
    ent = -1.0 * ent.sum()
    ent = ent / x.shape[0]
    return ent

def dist_to_nn_l1(x: np.ndarray) -> Tuple[float, float]:
    distances, nns = NN_L1.kneighbors(x, return_distance=True)
    avg = distances.mean()
    minn = distances.min()
    return avg, minn

def dist_to_nn_l2(x: np.ndarray) -> Tuple[float, float]:
    distances, nns = NN_L2.kneighbors(x, return_distance=True)
    avg = distances.mean()
    minn = distances.min()
    return avg, minn

log('running over all test samples in original trainset and OOD set...')
for i in tqdm(range(200)):
    # normal
    loader = get_single_img_dataloader(args.dataset, X_test, y_test, args.tta_size,
                                       pin_memory=device=='cuda', transform=tta_transforms, index=i, use_one_hot=False)
    (inputs, targets) = list(loader)[0]
    inputs, targets = inputs.to(device), targets.to(device)
    out = net(inputs)
    emb_vectors = out['embeddings'].detach().cpu().numpy()
    logits = out['logits'].detach().cpu().numpy()
    probs = out['probs'].detach().cpu().numpy()
    metrics['avg_l1_dist'][i, 0] = avg_l1_dist(emb_vectors)
    metrics['avg_l2_dist'][i, 0] = avg_l2_dist(emb_vectors)
    metrics['max_l1_dist'][i, 0] = max_l1_dist(emb_vectors)
    metrics['max_l2_dist'][i, 0] = max_l2_dist(emb_vectors)
    metrics['entropy'][i, 0] = entropy(probs)
    metrics['avg_l2_dist_to_nn'][i, 0], metrics['min_l2_dist_to_nn'][i, 0] = dist_to_nn_l2(emb_vectors)
    metrics['avg_l1_dist_to_nn'][i, 0], metrics['min_l1_dist_to_nn'][i, 0] = dist_to_nn_l1(emb_vectors)

    # ood
    loader = get_single_img_dataloader(args.ood_set, ood_X_test, ood_y_test, args.tta_size,
                                       pin_memory=device=='cuda', transform=ood_tta_transforms, index=i, use_one_hot=False)
    (inputs, targets) = list(loader)[0]
    inputs, targets = inputs.to(device), targets.to(device)
    out = net(inputs)
    emb_vectors = out['embeddings'].detach().cpu().numpy()
    logits = out['logits'].detach().cpu().numpy()
    probs = out['probs'].detach().cpu().numpy()
    metrics['avg_l1_dist'][i, 1] = avg_l1_dist(emb_vectors)
    metrics['avg_l2_dist'][i, 1] = avg_l2_dist(emb_vectors)
    metrics['max_l1_dist'][i, 1] = max_l1_dist(emb_vectors)
    metrics['max_l2_dist'][i, 1] = max_l2_dist(emb_vectors)
    metrics['entropy'][i, 1] = entropy(probs)
    metrics['avg_l2_dist_to_nn'][i, 1], metrics['min_l2_dist_to_nn'][i, 1] = dist_to_nn_l2(emb_vectors)
    metrics['avg_l1_dist_to_nn'][i, 1], metrics['min_l1_dist_to_nn'][i, 1] = dist_to_nn_l1(emb_vectors)

log('Dumping...')
with open(os.path.join(DUMP_DIR, 'metrics.pkl'), 'wb') as handle:
    pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

log('done')

# debug
# original images
# x_all = convert_tensor_to_image(X_train)
# ind = 0
# plt.imshow(x_all[ind])
# plt.show()


# ood
# x_all = convert_tensor_to_image(inputs.detach().cpu().numpy())
# for ind in np.arange(10):
#     plt.imshow(x_all[ind])
#     plt.show()

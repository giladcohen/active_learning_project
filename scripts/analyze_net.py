'''Train CIFAR10 with PyTorch.'''
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import time

from active_learning_project.models import *
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_loader_with_specific_inds, get_all_data_loader
from active_learning_project.datasets.selection_methods import select_random, update_inds, SelectionMethodFactory
from active_learning_project.utils import remove_substr_from_keys

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/active_learning/cifar10/full_dataset/full_lr_0.1_wd_0.00039_f_0.9_p_3_c_1_220120_03', type=str, help='checkpoint dir')
parser.add_argument('--distance_norm', default='L2', type=str, help='Distance norm. Can be [L1/L2/L_inf]')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/data/dataset/cifar10'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')



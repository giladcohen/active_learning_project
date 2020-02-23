'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import numpy as np

import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
from scipy.spatial.distance import pdist, cdist, squareform

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def remove_substr_from_keys(d: dict, substr):
    return {x.replace(substr, ''): v for x, v in d.items()}

# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
# TOTAL_BAR_LENGTH = 65.
# last_time = time.time()
# begin_time = last_time
# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.
#
#     cur_len = int(TOTAL_BAR_LENGTH*current/total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
#
#     sys.stdout.write(' [')
#     for i in range(cur_len):
#         sys.stdout.write('=')
#     sys.stdout.write('>')
#     for i in range(rest_len):
#         sys.stdout.write('.')
#     sys.stdout.write(']')
#
#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time
#
#     L = []
#     L.append('  Step: %s' % format_time(step_time))
#     L.append(' | Tot: %s' % format_time(tot_time))
#     if msg:
#         L.append(' | ' + msg)
#
#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#         sys.stdout.write(' ')
#
#     # Go back to the center of the bar.
#     for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#         sys.stdout.write('\b')
#     sys.stdout.write(' %d/%d ' % (current+1, total))
#
#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def pytorch_evaluate(net: nn.Module, data_loader: data.DataLoader, fetch_keys: list, to_tensor: bool=False) -> tuple:
    # Fetching inference outputs as numpy arrays
    batch_size = data_loader.batch_size
    num_samples = len(data_loader.dataset)
    batch_count = int(np.ceil(num_samples / batch_size))
    fetches_dict = {}
    fetches = []
    for key in fetch_keys:
        fetches_dict[key] = []

    net.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs_dict = net(inputs)
        for key in fetch_keys:
            fetches_dict[key].append(outputs_dict[key].data.cpu().detach().numpy())

    # stack variables together
    for key in fetch_keys:
        fetch = np.vstack(fetches_dict[key])
        if to_tensor:
            fetch = torch.as_tensor(fetch, device=torch.device(device))
        fetches.append(fetch)

    assert batch_idx + 1 == batch_count
    assert fetches[0].shape[0] == num_samples

    return tuple(fetches)

def validate_new_inds(selected_inds: list, inds_dict: dict):
    """Validate that selected_inds are not in the train/val_inds.
       On the other hand, validate that selected_inds are all in the unlabeled_inds
    """
    new_set       = set(selected_inds)
    train_set     = set(inds_dict['train_inds'])
    val_set       = set(inds_dict['val_inds'])
    unlabeled_set = set(inds_dict['unlabeled_inds'])

    assert len(new_set.intersection(train_set)) == 0, 'Some selected samples are already in the train set'
    assert len(new_set.intersection(val_set)) == 0, 'Some selected samples are already in the val set'
    assert new_set.issubset(unlabeled_set), 'All new selected indices must be in unlabeled_inds'

def convert_norm_str_to_p(norm: str):
    assert norm in ['L1', 'L2', 'L_inf']
    if norm in ['L1', 'L2']:
        p = int(norm[-1])
    else:
        p = np.inf
    return p

def calculate_dist_mat(embeddings: np.ndarray, norm: int) -> np.ndarray:
    """Returning a distance matrix from embeddings vector"""
    kwargs = {'p': norm}
    condensed_dist = pdist(embeddings, metric='minkowski', **kwargs)
    dist_mat = squareform(condensed_dist)
    return dist_mat

def calculate_dist_mat_2(A: np.ndarray, B: np.array, norm: int) -> np.ndarray:
    """Returning a distance matrix from embeddings vector"""
    kwargs = {'p': norm}
    dist_mat = cdist(A, B, metric='minkowski', **kwargs)
    return dist_mat

def boolean_string(s):
    # to use --use_bn True or --use_bn False in the shell. See:
    # https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# network funcs
def to_1d(x):
    return x.view(x.size(0), -1)

def activation_ratio(x):
    """
    :param x: feature map. tensor of size: [batch, feature_map_size, num_pix, num_pix], where num_pix=32/16/8/4
    :return: activation ratio per 2D conv kernel. size to return value: [batch, feature_map_size]
    """
    batch_size = x.size(0)
    is_1d = len(x.size()) == 2
    if is_1d:
        spatial_size = 1
        dim = 0
    else:
        spatial_size = x.size(2) * x.size(3)
        dim = (0, 2, 3)
    activated_sum = x.sign().sum(dim=dim)
    return activated_sum / (batch_size * spatial_size)

def activation_batch_ratio(x):
    """
    :param x: feature map. tensor of size: [batch, feature_map_size, num_pix, num_pix], where num_pix=32/16/8/4
    :return: activation ratio averaged on the batch, for every pixel. size to return value: [batch, feature_map_size]
    """
    batch_size = x.size(0)
    activated_sum = x.sign().sum()
    return activated_sum / batch_size

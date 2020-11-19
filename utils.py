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

# def get_np_data(data_loader: data.DataLoader) -> np.ndarray:
#     batch_size = data_loader.batch_size
#     for batch_idx, (inputs, targets) in enumerate(data_loader):
#         b = batch_idx * batch_size
#         e = b + targets.shape[0]
#         X[b:e] = inputs.cpu().numpy()

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

def activation_ratio_avg(x):
    """
    :param x: feature map. tensor of size: [batch, feature_map_size, num_pix, num_pix], where num_pix=32/16/8/4
    :return: average activation ratio on all 2D conv kernels. size to return value: [batch, feature_map_size]
    """
    return x.sign().mean()

def activation_L1_ratio(x):
    """
    :param x: feature map. tensor of size: [batch, feature_map_size, num_pix, num_pix], where num_pix=32/16/8/4
    :return: average L1 activation ratio on all 2D conv kernels. size to return value: [batch, feature_map_size]
    """
    batch_size = x.size(0)
    is_1d = len(x.size()) == 2
    if is_1d:
        spatial_size = 1
        dim = 0
    else:
        spatial_size = x.size(2) * x.size(3)
        dim = (0, 2, 3)

    activated_sum = (x[x > 0]).sqrt().sum()
    activated_sum = activated_sum / (batch_size * spatial_size)
    return torch.exp(-1.0 * activated_sum)

def activation_batch_ratio(x):
    """
    :param x: feature map. tensor of size: [batch, feature_map_size, num_pix, num_pix], where num_pix=32/16/8/4
    :return: activation ratio averaged on the batch, for every pixel. size to return value: scalar
    """
    batch_size = x.size(0)
    activated_sum = x.sign().sum()
    return activated_sum / batch_size

def convert_tensor_to_image(x: np.ndarray):
    """
    :param X: np.array of size (Batch, feature_dims, H, W)
    :return: X with (Batch, H, W, feature_dims) between 0:255, uint8
    """
    X = x.copy()
    X *= 255.0
    X = np.round(X)
    X = X.astype(np.uint8)
    X = np.transpose(X, [0, 2, 3, 1])
    return X

def majority_vote(x):
    return np.bincount(x).argmax()

def get_ensemble_paths(ensemble_dir):
    ensemble_subdirs = next(os.walk(ensemble_dir))[1]
    ensemble_subdirs.sort()
    ensemble_paths = []
    for j, dir in enumerate(ensemble_subdirs):  # for network j
        ensemble_paths.append(os.path.join(ensemble_dir, dir, 'ckpt.pth'))

    return ensemble_paths

def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)

def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)

def all_grads(y, x, create_graph=False):
    jac = torch.zeros_like(x)
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        jac[i] = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)[0][i]
        grad_y[i] = 0.
    return jac

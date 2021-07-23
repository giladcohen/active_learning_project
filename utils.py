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
from tqdm import tqdm
import pickle
import logging
from functools import wraps
import matplotlib.pyplot as plt

from numba import njit, jit

import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
import logging

import scipy
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics import roc_curve, auc, roc_auc_score
from active_learning_project.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101

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

def convert_image_to_tensor(x: np.ndarray):
    """
    :param X: np.array of size (Batch, H, W, feature_dims) between 0:255, uint8
    :return: X with (Batch, feature_dims, H, W) float between [0:1]
    """
    assert x.dtype == np.uint8
    X = x.copy()
    X = X.astype(np.float32)
    X /= 255.0
    X = np.transpose(X, [0, 3, 1, 2])
    return X

def majority_vote(x):
    return np.bincount(x).argmax()

def calc_prob(x: np.ndarray) -> np.ndarray:
    return scipy.special.softmax(x)

def calc_prob_wo_l(x: np.ndarray, l: int) -> np.ndarray:
    xx = x.copy()
    xx[l] = -np.inf
    return scipy.special.softmax(xx)

def get_is_adv_prob(preds: np.ndarray) -> np.ndarray:
    # first get all the probs:
    test_size, num_points, num_classes = preds.shape
    pi_mat = scipy.special.softmax(preds, axis=2)
    pil_mat = np.zeros((preds.shape) + (num_classes,)) # (test_size(k), num_points(n), #classes(l), #classes(i))
    for cls in range(num_classes):
        tmp_preds = preds.copy()
        tmp_preds[:, :, cls] = -np.inf
        pil_mat[:, :, cls] = scipy.special.softmax(tmp_preds, axis=2)

    p_is_adv = np.nan * np.ones(preds.shape)  #shape = (k, n, i) = (test_size, num_points, num_classes)
    for k in tqdm(range(preds.shape[0])):
        l = preds[k, 0].argmax()
        for n in range(num_points):
            for i in range(num_classes):
                if i != l:
                    pi = pi_mat[k, n, i]
                    pil = pil_mat[k, n, l, i]
                    p_is_adv[k, n, i] = (1 - pi) / (1 - pi + pil)

    return p_is_adv

def get_is_adv_prob_v2(preds):
    test_size, num_points, num_classes = preds.shape
    probs = scipy.special.softmax(preds, axis=2)
    probs_mean = probs.mean(axis=1)

    probs_wo_l = np.zeros((test_size, num_points, num_classes))
    for k in range(test_size):
        l = preds[k, 0].argmax()
        for n in range(num_points):
            probs_wo_l[k, n] = calc_prob_wo_l(preds[k, n], l)
    probs_wo_l_mean = probs_wo_l.mean(axis=1)

    p_is_adv = np.zeros(test_size)
    for k in range(test_size):
        l = preds[k, 0].argmax()
        p = 0.0
        for i in range(num_classes):
            if i != l:
                p += (1 - probs_mean[k, i]) / (1 - probs_mean[k, i] + probs_wo_l_mean[k, i])
        p_is_adv[k] = p / (num_classes - 1)

    return p_is_adv

def compute_roc(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

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

def add_feature(x, x1):
    """Adding feature x1 to x"""
    if x is None:
        x = x1
    else:
        x = np.concatenate((x, x1), axis=1)
    return x

def save_features(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=4)


class EMAOld(object):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

class EMA(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.params = self.shadow.keys()

    def __call__(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if name in self.params and param.requires_grad:
                    self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
                    param.data = self.shadow[name]

def set_logger(log_file):
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler(sys.stdout)]
                        )

def get_model(moder_str):
    if moder_str == 'resnet18':
        return ResNet18
    elif moder_str == 'resnet34':
        return ResNet34
    elif moder_str == 'resnet50':
        return ResNet50
    elif moder_str == 'resnet101':
        return ResNet101
    else:
        raise AssertionError("network {} is unknown".format(moder_str))


def reset_net(net):
    """This is not a robust solution and wont work for anything except core torch.nn layers:
    https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034/2
    """
    for layer in net.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def print_Linf_dists(X, X_test):
    logger = logging.getLogger()
    X_diff = (X - X_test).reshape(X.shape[0], -1)
    X_diff_abs = np.abs(X_diff)
    Linf_dist = X_diff_abs.max(axis=1)
    Linf_dist = Linf_dist[np.where(Linf_dist > 0.0)[0]]
    logger.info('The adversarial attacks distance: Max[L_inf]={}, E[L_inf]={}'.format(np.max(Linf_dist), np.mean(Linf_dist)))

def calc_attack_rate(y_preds: np.ndarray, y_orig_norm_preds: np.ndarray, y_gt: np.ndarray) -> float:
    """
    Args:
        y_preds: The adv image's final prediction after the defense method
        y_orig_norm_preds: The original image's predictions
        y_gt: The GT labels
        targeted: Whether or not the attack was targeted
    
    Returns: attack rate in %
    """
    f0_inds = []  # net_fail
    f1_inds = []  # net_succ
    f2_inds = []  # net_succ AND attack_flip

    for i in range(len(y_gt)):
        f1 = y_orig_norm_preds[i] == y_gt[i]
        f2 = f1 and y_preds[i] != y_orig_norm_preds[i]
        if f1:
            f1_inds.append(i)
        else:
            f0_inds.append(i)
        if f2:
            f2_inds.append(i)

    attack_rate = len(f2_inds) / len(f1_inds)
    return attack_rate


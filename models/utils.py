import numpy as np

def get_strides(dataset: str):
    if dataset in ['cifar10', 'cifar100', 'svhn']:
        strides = [1, 2, 2, 2]
    elif dataset == 'tiny_imagenet':
        strides = [2, 2, 2, 2]
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))
    return strides

def get_conv1_params(dataset: str):
    if dataset in ['cifar10', 'cifar100', 'svhn']:
        conv1 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
    elif dataset == 'tiny_imagenet':
        conv1 = {'kernel_size': 7, 'stride': 1, 'padding': 3}
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))
    return conv1

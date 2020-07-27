"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import torch
import numpy as np
import os

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Subset
from active_learning_project.datasets.my_svhn import MySVHN
import matplotlib.pyplot as plt

BASE_DATASET_DIR = '/Users/giladcohen/data/dataset'
def dataset_factory(dataset):
    if dataset == 'cifar10':
        data_dir = os.path.join(BASE_DATASET_DIR, 'cifar10')
        database = datasets.CIFAR10
    elif dataset == 'cifar100':
        data_dir = os.path.join(BASE_DATASET_DIR, 'cifar100')
        database = datasets.CIFAR100
    elif dataset == 'svhn':
        data_dir = os.path.join(BASE_DATASET_DIR, 'svhn')
        database = MySVHN
    else:
        raise AssertionError('dataset {} is not supported'.format(dataset))

    if 'cifar' in dataset:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return data_dir, database, train_transform, test_transform


def get_train_valid_loader(dataset,
                           batch_size,
                           rand_gen,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 train_dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - dataset: name of the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - rand_gen: rand_gen for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the train_dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    data_dir, database, train_transform, test_transform = dataset_factory(dataset)

    # load the dataset
    train_dataset = database(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = database(
        root=data_dir, train=True,
        download=True, transform=test_transform,
    )

    num_train_val = len(train_dataset)
    num_val       = int(np.floor(valid_size * num_train_val))
    indices = list(range(num_train_val))

    train_idx, val_idx = \
        train_test_split(indices, test_size=num_val, random_state=rand_gen, shuffle=shuffle, stratify=train_dataset.targets)
    train_idx.sort()
    val_idx.sort()

    train_dataset.data = train_dataset.data[train_idx]
    train_dataset.targets = np.asarray(train_dataset.targets)[train_idx]
    valid_dataset.data = valid_dataset.data[val_idx]
    valid_dataset.targets = np.asarray(valid_dataset.targets)[val_idx]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, train_idx, val_idx

def get_loader_with_specific_inds(dataset,
                                  batch_size,
                                  is_training,
                                  indices,
                                  num_workers=4,
                                  pin_memory=False):
    """
    Same like get_train_valid_loader but with exact indices for training and validation
    """
    data_dir, database, train_transform, test_transform = dataset_factory(dataset)

    if is_training:
        transform = train_transform
    else:
        transform = test_transform

    # load the dataset
    dataset = database(
        root=data_dir, train=True,
        download=True, transform=transform,
    )
    dataset.data = dataset.data[indices]
    dataset.targets = np.asarray(dataset.targets)[indices]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=is_training,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return loader

def get_all_data_loader(dataset,
                   batch_size,
                   num_workers=4,
                   pin_memory=False):
    """
    Same like get_train_valid_loader but with exact indices for training and validation
    """
    data_dir, database, train_transform, test_transform = dataset_factory(dataset)

    # load the dataset
    dataset = database(
        root=data_dir, train=True,
        download=True, transform=test_transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return loader


def get_test_loader(dataset,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - dataset: name of the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """

    data_dir, database, train_transform, test_transform = dataset_factory(dataset)

    dataset = database(
        root=data_dir, train=False,
        download=True, transform=test_transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def get_normalized_tensor(loader: torch.utils.data.DataLoader, batch_size=100):
    """ Returning a normalized tensor"""
    size = len(loader.dataset)
    X = -1.0 * np.ones(shape=(size, 3, 32, 32), dtype=np.float32)
    for batch_idx, (inputs, targets) in enumerate(loader):
        b = batch_idx * batch_size
        e = b + targets.shape[0]
        X[b:e] = inputs.cpu().numpy()

    return X

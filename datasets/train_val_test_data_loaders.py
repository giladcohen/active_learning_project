"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Subset


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
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
    - data_dir: path directory to the train_dataset.
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

    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train_val = len(train_dataset)
    num_val       = int(np.floor(valid_size * num_train_val))
    indices = list(range(num_train_val))

    train_idx, val_idx = \
        train_test_split(indices, test_size=num_val, random_state=rand_gen, shuffle=shuffle, stratify=train_dataset.targets)
    train_idx.sort()
    val_idx.sort()

    # normalize
    train_dataset.data = train_dataset.data / 255.0
    valid_dataset.data = valid_dataset.data / 255.0

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

def get_loader_with_specific_inds(data_dir,
                                  batch_size,
                                  is_training,
                                  indices,
                                  num_workers=4,
                                  pin_memory=False):
    """
    Same like get_train_valid_loader but with exact indices for training and validation
    """
    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )

    # define transforms
    if is_training:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ])

    # load the dataset
    dataset = datasets.CIFAR10(
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

def get_all_data_loader(data_dir,
                   batch_size,
                   num_workers=4,
                   pin_memory=False):
    """
    Same like get_train_valid_loader but with exact indices for training and validation
    """
    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )

    # define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize,
    ])

    # load the dataset
    dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )
    dataset.data = dataset.data / 255.0

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return loader


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    #     # WAS:
    #     # mean=[0.485, 0.456, 0.406],
    #     # std=[0.229, 0.224, 0.225],
    #     # but changed to match the validation scaling
    # )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )
    dataset.data = dataset.data / 255.0

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
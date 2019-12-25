import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torch.utils.data.dataset import Subset
import torch.nn as nn
import torch.utils.data as data

rand_gen = np.random.RandomState(12345)
DATA_ROOT = '/data/dataset/cifar10'

def init_select(selection_size: int, dataset_size: int) -> list:
    """selecting selection_size insidces out of dataset_size"""
    selected_inds = rand_gen.choice(dataset_size, selection_size, replace=False)
    selected_inds.sort()
    selected_inds = selected_inds.tolist()
    return selected_inds

def update_inds(train_inds: list, val_inds: list, new_inds: list, val_perc=5.0) -> None:
    """Adding to train_inds and val_inds new indices from new_inds
    How many indices are going to the train and how many to the val?
    """
    new_inds = np.asarray(new_inds, dtype=np.int32)
    new_val_num = int(np.floor(len(new_inds) * val_perc * 0.01))

    dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True)
    subset = Subset(dataset, new_inds)
    targets = np.asarray(subset.dataset.targets)[subset.indices]

    new_train_inds, new_val_inds = \
        train_test_split(new_inds, test_size=new_val_num, random_state=rand_gen, shuffle=True, stratify=targets)

    new_train_inds = new_train_inds.tolist()
    new_val_inds   = new_val_inds.tolist()

    train_inds += new_train_inds
    val_inds   += new_val_inds

    train_inds.sort()
    val_inds.sort()

def select_random(net: nn.Module, dataset: data.Dataset, selection_size: int, unlabeled_inds: list):
    selected_inds = rand_gen.choice(np.asarray(unlabeled_inds), selection_size, replace=False)
    selected_inds.sort()
    selected_inds = selected_inds.tolist()
    return selected_inds

class SelectionMethodFactory(object):

    def config(self, name):
        if name == 'random':
            return select_random
        raise AssertionError('not selection method named {}'.format(name))

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torch.utils.data.dataset import Subset
import torch.nn as nn
import torch.utils.data as data
from active_learning_project.utils import pytorch_evaluate
import time

rand_gen = np.random.RandomState(int(time.time()))
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

def uncertainty_score(y_pred_dnn):
    """
    Calculates the uncertainty score for y_pred_dnn
    :param y_pred_dnn: np.float32 array of the DNN probability
    :return: uncertainty score for every vector
    """
    return 1.0 - y_pred_dnn.max(axis=1)

def select_random(net: nn.Module, data_loader: data.DataLoader, selection_size: int, inds_dict: dict):
    unlabeled_inds = inds_dict['unlabeled_inds']
    selected_inds = rand_gen.choice(np.asarray(unlabeled_inds), selection_size, replace=False)
    selected_inds.sort()
    selected_inds = selected_inds.tolist()
    return selected_inds

def select_confidence(net: nn.Module, data_loader: data.DataLoader, selection_size: int, inds_dict: dict):
    (logits, ) = pytorch_evaluate(net, data_loader, fetch_keys=['logits'], to_tensor=True)
    pred_probs = nn.Softmax(logits)
    pred_probs = pred_probs[inds_dict['unlabeled_inds']]
    uncertainties = uncertainty_score(pred_probs)
    best_indices_relative = uncertainties.argsort()[-selection_size:]
    best_indices = np.take(inds_dict['unlabeled_inds'], best_indices_relative)
    best_indices.tolist()
    best_indices.sort()
    return best_indices

class SelectionMethodFactory(object):

    def config(self, name):
        if name == 'random':
            return select_random
        elif name == 'confidence':
            return select_confidence
        raise AssertionError('not selection method named {}'.format(name))

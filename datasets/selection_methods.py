import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torch.utils.data.dataset import Subset
import torch.nn as nn
import torch.utils.data as data
from active_learning_project.utils import pytorch_evaluate, validate_new_inds, calculate_dist_mat, calculate_dist_mat_2
import time
import torch.nn.functional as F
from active_learning_project.utils import convert_norm_str_to_p
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

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

def select_random(net: nn.Module, data_loader: data.DataLoader, inds_dict: dict, cfg: dict=None):
    selected_inds = rand_gen.choice(np.asarray(inds_dict['unlabeled_inds']), cfg['selection_size'], replace=False)
    selected_inds.sort()
    selected_inds = selected_inds.tolist()
    validate_new_inds(selected_inds, inds_dict)
    return selected_inds

def select_confidence(net: nn.Module, data_loader: data.DataLoader, inds_dict: dict, cfg: dict=None):
    (logits, ) = pytorch_evaluate(net, data_loader, fetch_keys=['logits'], to_tensor=True)
    pred_probs = F.softmax(logits).cpu().detach().numpy()
    pred_probs = pred_probs[inds_dict['unlabeled_inds']]
    uncertainties = uncertainty_score(pred_probs)
    selected_inds_relative = uncertainties.argsort()[-cfg['selection_size']:]
    selected_inds = np.take(inds_dict['unlabeled_inds'], selected_inds_relative)
    selected_inds = selected_inds.tolist()
    selected_inds.sort()
    validate_new_inds(selected_inds, inds_dict)
    return selected_inds

def select_farthest(net: nn.Module, data_loader: data.DataLoader, inds_dict: dict, cfg: dict=None):
    (embeddings, ) = pytorch_evaluate(net, data_loader, fetch_keys=['embeddings'], to_tensor=False)
    norm = convert_norm_str_to_p(cfg['distance_norm'])

    # print('Constructing the distsnce matrix...')
    # dist_mat = calculate_dist_map(embeddings, norm)
    # dist_mat_tmp = dist_mat[untaken_inds]
    # dist_mat_tmp = dist_mat_tmp[:, taken_inds]

    taken_inds = inds_dict['train_inds']
    if cfg['include_val_as_train']:
        taken_inds += inds_dict['val_inds']
    untaken_inds = inds_dict['unlabeled_inds'].copy()

    # knn = NearestNeighbors(
    #     n_neighbors=len(taken_inds),
    #     p=norm,
    #     algorithm='brute',
    #     n_jobs=20
    # )
    # knn.fit(embeddings[taken_inds])
    # dist_mat, _ = knn.kneighbors(embeddings[untaken_inds], return_distance=True)

    dist_mat = calculate_dist_mat_2(embeddings[untaken_inds], embeddings[taken_inds], norm)

    selected_inds = []
    for i in tqdm(range(cfg['selection_size'])):
        min_dists = dist_mat.min(axis=1)
        selected_ind_relative = min_dists.argmax()
        selected_ind = np.take(untaken_inds, selected_ind_relative)

        # update selected inds:
        selected_inds.append(selected_ind)
        assert selected_ind not in taken_inds
        taken_inds.append(selected_ind)
        untaken_inds.remove(selected_ind)

        # update dist_mat
        # first, removing the row that correspond to the untaken index
        dist_mat = dist_mat[untaken_inds]
        # next, we need to add the distance of all the (remaining) untaken inds to the newest selected_ind
        # to that end, just calculate the distance from all the untaken inds to the freshly new taken ind
        new_dists = calculate_dist_mat_2(embeddings[untaken_inds], embeddings[np.newaxis, selected_ind], norm)
        dist_mat = np.hstack((dist_mat, new_dists))

    assert len(selected_inds) == cfg['selection_size']
    selected_inds.sort()
    validate_new_inds(selected_inds, inds_dict)
    return selected_inds

class SelectionMethodFactory(object):

    def config(self, name):
        if name == 'random':
            return select_random
        elif name == 'confidence':
            return select_confidence
        elif name == 'farthest':
            return select_farthest
        raise AssertionError('not selection method named {}'.format(name))

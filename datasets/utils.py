import numpy as np
import os

def get_mini_dataset_inds(dataset: str):
    val_path = os.path.join('/data/gilad/logs/adv_robustness', dataset, 'test_val_inds.npy')
    test_path = os.path.join('/data/gilad/logs/adv_robustness', dataset, 'test_test_inds.npy')
    val_inds = np.load(val_path)
    test_inds = np.load(test_path)
    return val_inds, test_inds

def get_ensemble_dir(dataset: str, net: str):
    return os.path.join('/data/gilad/logs/adv_robustness', dataset, net, 'regular')

def get_tta_dir(checkpoint_dir, tta_dir, attack_dir):
    if attack_dir == '':
        return os.path.join(checkpoint_dir, tta_dir)
    else:
        return os.path.join(checkpoint_dir, attack_dir, tta_dir)

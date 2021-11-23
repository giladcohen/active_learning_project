import numpy as np
import os

def get_dataset_inds(dataset: str):
    val_path = os.path.join('/data/gilad/logs/adv_robustness', dataset, 'test_val_inds.npy')
    test_path = os.path.join('/data/gilad/logs/adv_robustness', dataset, 'test_test_inds.npy')
    val_inds = np.load(val_path)
    test_inds = np.load(test_path)
    return val_inds, test_inds

def get_mini_dataset_inds(dataset: str):
    val_path = os.path.join('/data/gilad/logs/adv_robustness', dataset, 'mini_test_val_inds.npy')
    test_path = os.path.join('/data/gilad/logs/adv_robustness', dataset, 'mini_test_test_inds.npy')
    val_inds = np.load(val_path)
    test_inds = np.load(test_path)
    return val_inds, test_inds

def get_ensemble_dir(dataset: str, net: str):
    return os.path.join('/data/gilad/logs/adv_robustness', dataset, net, 'regular')

def get_dump_dir(checkpoint_dir, tta_dir, attack_dir):
    if attack_dir == '':
        return os.path.join(checkpoint_dir, 'normal', tta_dir)
    else:
        return os.path.join(checkpoint_dir, attack_dir, tta_dir)

def get_boundary_val_test_inds(dataset):
    mini_val_inds, mini_test_inds = get_mini_dataset_inds(dataset)
    mini_inds = np.concatenate((mini_val_inds, mini_test_inds))
    mini_inds.sort()
    boundary_val_inds = np.asarray([i for i, ind in enumerate(mini_inds) if ind in mini_val_inds])
    boundary_test_inds = np.asarray([i for i, ind in enumerate(mini_inds) if ind in mini_test_inds])
    return boundary_val_inds, boundary_test_inds

def get_attack_inds(dataset, attack, is_vat, is_resnet101):
    # ugly hack:
    if attack == 'cw_targeted' and is_vat and is_resnet101 and dataset == 'tiny_imagenet':
        x_inds, gt_inds = 'mini_for_boundary', 'mini'
    elif attack in ['fgsm', 'jsma', 'pgd', 'deepfool', 'cw', 'cw_Linf', 'square']:
        x_inds, gt_inds = 'test', 'test'
    elif attack == 'boundary':
        x_inds, gt_inds = 'mini_for_boundary', 'mini'
    elif attack == 'whitebox_pgd':
        if is_vat:
            x_inds, gt_inds = 'none', 'test'
        else:
            x_inds, gt_inds = 'test', 'test'
    elif attack in ['bpda', 'adaptive_square', 'adaptive_boundary']:
        x_inds, gt_inds = 'none', 'mini'
    else:
        raise AssertionError('cannot find attack inds for attack: {}, is_vat: {}'.format(attack, is_vat))

    val_inds, test_inds = get_dataset_inds(dataset)
    mini_val_inds, mini_test_inds = get_mini_dataset_inds(dataset)
    boundary_val_inds, boundary_test_inds = get_boundary_val_test_inds(dataset)

    if x_inds == 'test':
        x_inds = test_inds  # for almost all attacks
    elif x_inds == 'mini_for_boundary':
        x_inds = boundary_test_inds  # for the boundary
    elif x_inds == 'none':
        x_inds = None
    else:
        raise AssertionError('How did we get here?')

    if gt_inds == 'test':
        gt_inds = test_inds  # for almost all attacks
    elif gt_inds == 'mini':
        gt_inds = mini_test_inds  # for quick attacks: Boundary, BPDA, adaptive_square, adaptive_boundary and some thitebox_pgd
    else:
        raise AssertionError('How did we get here?')

    return x_inds, gt_inds

"""Plot ablation study of tta_size for:
 CIFAR10, CIFAR100, SVHN,
 attack: PGD [plot1] and CW_Linf [plot2]"""

import os
import matplotlib.pyplot as plt
import sys
import numpy as np

def attack_to_dir(attack: str):
    if attack == 'PGD':
        return 'pgd_targeted'
    elif attack == 'Deepfool':
        return 'deepfool'
    elif attack == 'CW':
        return 'cw_targeted_Linf_eps_0.031'

def get_acc_from_log(file: str):
    acc = None
    with open(file, 'r') as f:
        for line in f:
            if 'INFO Test accuracy:' in line:
                acc = float(line.split('accuracy: ')[1].split('%')[0])
    assert acc is not None
    return acc


num_experiments = 5
CHECKPOINT_ROOT = '/data/gilad/logs/adv_robustness'
datasets = ['cifar10', 'cifar100', 'svhn']
attacks = ['PGD', 'Deepfool', 'CW']
tta_size_vec = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

data = {}
for dataset in datasets:
    data[dataset] = {}
    for attack in attacks:
        data[dataset][attack] = {}
        for tta_size in tta_size_vec:
            data[dataset][attack][tta_size] = []
            for n in range(1, num_experiments + 1):
                file = os.path.join(CHECKPOINT_ROOT, dataset, 'resnet34', 'regular', 'resnet34_00', attack_to_dir(attack), 'tta_size_{}_exp{}'.format(tta_size, n), 'log.log')
                if (dataset == 'cifar100' and attack == 'CW' and tta_size == 1024 and n == 3) or \
                   (dataset == 'svhn' and attack == 'CW' and tta_size == 512 and n == 3) or \
                   (dataset == 'svhn' and attack == 'CW' and tta_size == 1024 and n == 2) or \
                   (dataset == 'svhn' and attack == 'CW' and tta_size == 1024 and n == 3):
                    continue
                else:
                    acc = get_acc_from_log(file)
                    acc = np.round(acc, 2)
                    data[dataset][attack][tta_size].append(acc)
            data[dataset][attack][tta_size] = np.asarray(data[dataset][attack][tta_size])






# data['cifar10'] = {}
# data['cifar100'] = {}
# data['svhn'] = {}
#
# data['cifar10']['pgd']  = [83.60, 85.72, 86.08, 86.88, 86.48, 87.12, 86.84, 86.80, 86.80]
# data['cifar100']['pgd'] = [58.08, 60.52, 62.00, 62.24, 62.92, 62.56, 62.88, 63.00, 62.84]
# data['svhn']['pgd']     = [86.88, 87.96, 88.88, 89.56, 89.68, 89.60, 89.80, 89.64, 89.68]
#
#
#
# data['cifar10']['cw']  = [77.56, 80.4, 80.52, 80.72, 81.36, 81.56, ]

"""
KATANA: Knock-out Adversaries via Test-time AugmentatioN Aggregation.
Plot accuracy and adversarial accuracy for: CIFAR10/CIFAR100/SVHN/tiny_imagenet
for: resnet34/resnet50/resnet101
for methods: simple/ensemble/TRADES/TTA+RF
database is defined as: data[dataset][arch][attack]. attack='' means normal (unattacked) test samples.
"""

import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
CHECKPOINT_ROOT = '/data/gilad/logs/adv_robustness'
datasets = ['CIFAR10', 'CIFAR100', 'SVHN', 'Tiny-Imagenet']
archs = ['Resnet34']  #, 'Resnet50', 'Resnet101']
attacks = ['', 'FGSM1', 'FGSM2', 'JSMA', 'PGD1', 'PGD2', 'Deepfool', 'CW_L2', 'CW_Linf']
methods = ['simple', 'ensemble', 'TRADES', 'TTA', 'KATANA']  # 'TRADES+TTA', 'TRADES+KATANA']
data = {}

def dataset_to_dir(dataset: str):
    if dataset == 'CIFAR10':
        return 'cifar10'
    elif dataset == 'CIFAR100':
        return 'cifar100'
    elif dataset == 'SVHN':
        return 'svhn'
    elif dataset == 'Tiny-Imagenet':
        return 'tiny_imagenet'

def arch_to_dir(arch: str):
    return arch.lower()

def attack_to_dir(attack: str):
    if attack == '':
        return 'normal'
    elif attack == 'FGSM1':
        return 'fgsm_targeted'
    elif attack == 'FGSM2':
        return 'fgsm_targeted_eps_0.031'
    elif attack == 'JSMA':
        return 'jsma_targeted'
    elif attack == 'PGD1':
        return 'pgd_targeted'
    elif attack == 'PGD2':
        return 'pgd_targeted_eps_0.031'
    elif attack == 'Deepfool':
        return 'deepfool'
    elif attack == 'CW_L2':
        return 'cw_targeted'
    elif attack == 'CW_Linf':
        return 'cw_targeted_Linf_eps_0.031'

def method_to_dir(method:str):
    if method == 'simple':
        return 'simple'
    elif method == 'ensemble':
        return 'ensemble'
    elif method == 'TRADES':
        return 'simple'
    elif method == 'TTA':
        return 'tta'
    elif method == 'KATANA':
        return 'random_forest'
    elif method == 'TRADES+TTA':
        return 'tta'
    elif method == 'TRADES+KATANA':
        return 'random_forest'

def get_log(dataset: str, arch: str, attack: str, method: str):
    path = os.path.join(CHECKPOINT_ROOT, dataset_to_dir(dataset))
    path = os.path.join(path, arch_to_dir(arch))
    if 'TRADES' in method:
        path = os.path.join(path, 'adv_robust_trades')
    else:
        path = os.path.join(path, 'regular', arch_to_dir(arch) + '_00')
    path = os.path.join(path, attack_to_dir(attack))
    path = os.path.join(path, method_to_dir(method))
    path = os.path.join(path, 'log.log')
    return path

def get_simple_acc_from_log(log: str):
    ret = None
    with open(log, 'r') as f:
        for line in f:
            if 'INFO Normal test accuracy:' in line:
                ret = float(line.split('accuracy: ')[1].split('%')[0])
    assert ret is not None
    ret = np.round(ret, 2)
    return ret

def get_normal_katana_acc_from_log(log: str):
    ret = None
    with open(log, 'r') as f:
        for line in f:
            if 'INFO New normal test accuracy of random_forest:' in line:
                ret = float(line.split('random_forest: ')[1].split('%')[0])
    assert ret is not None
    ret = np.round(ret, 2)
    return ret

def get_acc_from_log(log: str):
    ret = None
    with open(log, 'r') as f:
        for line in f:
            if 'INFO Test accuracy:' in line:
                ret = float(line.split('accuracy: ')[1].split('%')[0])
    assert ret is not None
    ret = np.round(ret, 2)
    return ret

def get_attack_success_from_log(log: str):
    ret = None
    with open(log, 'r') as f:
        for line in f:
            if 'INFO attack success rate:' in line:
                ret = float(line.split('success rate: ')[1].split('%')[0])
    assert ret is not None
    ret = np.round(ret, 3)
    return ret

def get_avg_attack_norm_from_log(log: str):
    ret = None
    with open(log, 'r') as f:
        for line in f:
            if 'INFO The adversarial attacks distance:' in line:
                ret = float(line.split('E[L_inf]=')[1].split('%')[0])
    assert ret is not None
    ret = np.round(ret, 4)
    return ret

for dataset in datasets:
    data[dataset] = {}
    for arch in archs:
        data[dataset][arch] = {}
        for attack in attacks:
            data[dataset][arch][attack] = {}
            for method in methods:
                data[dataset][arch][attack][method] = \
                    {'normal_acc': np.nan, 'adv_acc': np.nan, 'attack_rate': np.nan, 'avg_attack_norm': np.nan}
                is_attacked = attack != ''
                is_katana = 'KATANA' in method
                if not is_attacked and is_katana:  # KATANA works only for attacked
                    continue

                log = get_log(dataset, arch, attack, method)
                print('for {}/{}/{}/{} , log: {}, we got:'.format(dataset, arch, attack, method, log))

                if not is_attacked and method in ['simple', 'TRADES']:
                    data[dataset][arch][attack][method]['normal_acc'] = get_simple_acc_from_log(log)
                elif not is_attacked:
                    data[dataset][arch][attack][method]['normal_acc'] = get_acc_from_log(log)
                else:
                    data[dataset][arch][attack][method]['adv_acc'] = get_acc_from_log(log)
                    data[dataset][arch][attack][method]['attack_rate'] = get_attack_success_from_log(log)
                    data[dataset][arch][attack][method]['avg_attack_norm'] = get_avg_attack_norm_from_log(log)
                if is_katana:
                    data[dataset][arch][attack][method]['normal_acc'] = get_normal_katana_acc_from_log(log)

                print(data[dataset][arch][attack][method])
                print('cool')


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
from typing import Dict
import seaborn as sns
sns.set_style("whitegrid")
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
CHECKPOINT_ROOT = '/data/gilad/logs/adv_robustness'
datasets = ['CIFAR-10', 'CIFAR-100', 'SVHN', 'Tiny-Imagenet']
archs = ['Resnet34']  #, 'Resnet50', 'Resnet101']
attacks = ['', 'FGSM1', 'FGSM2', 'JSMA', 'PGD1', 'PGD2', 'Deepfool', 'CW_L2', 'CW_Linf', 'Square', 'Boundary'
           ,'FGSM_WB', 'PGD_WB', 'A-Square', 'BPDA']
methods = ['simple', 'ensemble', 'TRADES', 'VAT', 'TTA', 'RF', 'TRADES+TTA', 'VAT+TTA', 'TRADES+RF', 'VAT+RF']
data = {}

def dataset_to_dir(dataset: str):
    if dataset == 'CIFAR-10':
        return 'cifar10'
    elif dataset == 'CIFAR-100':
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
    elif attack == 'Square':
        return 'square'
    elif attack == 'Boundary':
        return 'boundary_targeted'
    elif attack == 'FGSM_WB':
        return 'whitebox_pgd_targeted_iters_1_ttas_256'
    elif attack == 'PGD_WB':
        return 'whitebox_pgd_targeted_iters_10_ttas_25'
    elif attack == 'A-Square':
        return 'adaptive_square'
    elif attack == 'BPDA':
        return 'bpda_targeted'

def method_to_dir(method:str):
    if method in ['simple', 'TRADES', 'VAT']:
        return 'simple'
    elif method == 'ensemble':
        return 'ensemble'
    elif 'TTA' in method:
        return 'tta'
    elif method in ['RF', 'TRADES+RF', 'VAT+RF']:
        return 'random_forest_global'

def get_log(dataset: str, arch: str, attack: str, method: str):
    path = os.path.join(CHECKPOINT_ROOT, dataset_to_dir(dataset))
    path = os.path.join(path, arch_to_dir(arch))
    if 'TRADES' in method:
        path = os.path.join(path, 'adv_robust_trades')
    elif 'VAT' in method:
        path = os.path.join(path, 'adv_robust_vat')
    else:
        path = os.path.join(path, 'regular', arch_to_dir(arch) + '_00')
    attack_path = os.path.join(path, attack_to_dir(attack))

    path = os.path.join(attack_path, method_to_dir(method))
    if method in ['RF', 'TRADES+RF', 'VAT+RF']:
        path_bu = os.path.join(attack_path, 'random_forest')
    else:
        path_bu = os.path.join(attack_path, 'STAM')

    if not os.path.exists(path) and os.path.exists(path_bu):
        path = path_bu

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

def print_accs(x: Dict):
    vals = []
    vals.append(x['']['acc'])
    vals.append(x['FGSM1']['acc'])
    vals.append(x['FGSM2']['acc'])
    vals.append(x['JSMA']['acc'])
    vals.append(x['PGD1']['acc'])
    vals.append(x['PGD2']['acc'])
    vals.append(x['Deepfool']['acc'])
    vals.append(x['CW_L2']['acc'])
    vals.append(x['CW_Linf']['acc'])
    vals.append(x['Square']['acc'])
    vals.append(x['Boundary']['acc'])
    vals.append(x['A-Square']['acc'])
    # if 'FGSM_WB' in x:
    vals.append(x['FGSM_WB']['acc'])
    # if 'PGD_WB' in x:
    vals.append(x['PGD_WB']['acc'])
    vals.append(x['BPDA']['acc'])
    vals = np.asarray(vals)
    print(vals)

def print_white(x: Dict):
    vals = []
    vals.append(x['FGSM2']['acc'])
    vals.append(x['FGSM_WB']['acc'])
    vals.append(x['PGD2']['acc'])
    vals.append(x['PGD_WB']['acc'])
    vals = np.asarray(vals)
    print(vals)

def print_fast(x: Dict):
    vals = []
    vals.append(x['FGSM2']['acc'])
    vals.append(x['FGSM_WB']['acc'])
    vals.append(x['PGD2']['acc'])
    vals.append(x['PGD_WB']['acc'])
    vals.append(x['Square']['acc'])
    vals.append(x['A-Square']['acc'])
    vals.append(x['BPDA']['acc'])
    vals = np.asarray(vals)
    print(vals)

for dataset in datasets:
    data[dataset] = {}
    for arch in archs:
        data[dataset][arch] = {}
        for method in methods:
            data[dataset][arch][method] = {}
            for attack in attacks:
                # if attack in ['FGSM_WB', 'PGD_WB'] and arch != 'Resnet34':
                #     continue
                data[dataset][arch][method][attack] = {}
                data[dataset][arch][method][attack] = {'acc': np.nan, 'attack_rate': np.nan, 'avg_attack_norm': np.nan}
                is_attacked = attack != ''

                log = get_log(dataset, arch, attack, method)
                print('for {}/{}/{}/{} , log: {}, we got:'.format(dataset, arch, attack, method, log))

                if not is_attacked and method in ['simple', 'TRADES', 'VAT']:
                    data[dataset][arch][method][attack]['acc'] = get_simple_acc_from_log(log)
                elif not is_attacked:
                    data[dataset][arch][method][attack]['acc'] = get_acc_from_log(log)
                else:
                    data[dataset][arch][method][attack]['acc'] = get_acc_from_log(log)
                    data[dataset][arch][method][attack]['attack_rate'] = get_attack_success_from_log(log)
                    data[dataset][arch][method][attack]['avg_attack_norm'] = get_avg_attack_norm_from_log(log)

                # print(data[dataset][arch][attack][method])

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

def dataset_to_dir(dataset: str):
    if dataset == 'CIFAR10':
        return 'cifar10'
    elif dataset == 'CIFAR100':
        return 'cifar100'
    elif dataset == 'SVHN':
        return 'svhn'
    elif dataset == 'Tiny-Imagenet':
        return 'tiny_imagenet'

def attack_fo_dir(attack: str):
    if attack == 'FGSM1':
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




datasets = ['CIFAR10', 'CIFAR100', 'SVHN', 'Tiny-Imagenet']
attacks = ['FGSM1', 'FGSM2', 'JSMA', 'PGD1', 'PGD2', 'Deepfool', 'CW_L2', 'CW_Linf']
method = ['simple', 'ensemble', 'TRADES', 'TTA', 'KATANA', 'TRADES+TTA', 'TARDES+KATANA']







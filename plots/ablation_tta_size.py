"""Plot ablation study of tta_size for:
 CIFAR10, CIFAR100, SVHN,
 attack: PGD [plot1] and CW_Linf [plot2]"""

import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")


def attack_to_dir(attack: str):
    if attack == 'PGD':
        return 'pgd_targeted'
    elif attack == 'Deepfool':
        return 'deepfool'
    elif attack == 'CW':
        return 'cw_targeted_Linf_eps_0.031'

def get_acc_from_log(file: str):
    acc = np.nan
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
tta_size_ext_vec = []
data_ext = {}
for size in tta_size_vec:
    tta_size_ext_vec.extend([size] * num_experiments)

data = {}
for dataset in datasets:
    data[dataset] = {}
    for attack in attacks:
        data[dataset][attack] = {}
        for tta_size in tta_size_vec:
            data[dataset][attack][tta_size] = []
            for n in range(1, num_experiments + 1):
                file = os.path.join(CHECKPOINT_ROOT, dataset, 'resnet34', 'regular', 'resnet34_00', attack_to_dir(attack), 'tta_size_{}_exp{}'.format(tta_size, n), 'log.log')
                # if (dataset == 'cifar100' and attack == 'CW' and tta_size == 1024 and n == 3) or \
                #    (dataset == 'svhn' and attack == 'CW' and tta_size == 512 and n == 3) or \
                #    (dataset == 'svhn' and attack == 'CW' and tta_size == 1024 and n == 2) or \
                #    (dataset == 'svhn' and attack == 'CW' and tta_size == 1024 and n == 3):
                #     continue
                # else:
                acc = get_acc_from_log(file)
                acc = np.round(acc, 2)
                data[dataset][attack][tta_size].append(acc)
            data[dataset][attack][tta_size] = np.asarray(data[dataset][attack][tta_size])

for dataset in datasets:
    for attack in attacks:
        data_ext[attack] = []
        for size in tta_size_vec:
            data_ext[attack].extend(data[dataset][attack][size])

    d = {'tta_size': tta_size_ext_vec}
    for attack in attacks:
        d.update({attack: data_ext[attack]})
    df = pd.DataFrame(d)

    g = sns.lineplot(x='tta_size', y='value', hue='variable', data=pd.melt(df, ['tta_size']), ci='sd')
    g.set(xscale='log', xlabel='TTA size', ylabel='Accuracy [%]', title=dataset)
    g.set_xticks(tta_size_vec)
    g.set_xticklabels(tta_size_vec)
    plt.show()
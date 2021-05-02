import numpy as np
import json
import os
import argparse
import sys
import pickle
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

parser = argparse.ArgumentParser(description='OOD metrics plots')
parser.add_argument('--checkpoint_dir',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00',
                    type=str, help='checkpoint dir')
parser.add_argument('--ood_set', default='cifar100', type=str, help='OOD set: cifar10, cifar100, or svhn')
parser.add_argument('--dump_dir', default='dump_only_200', type=str, help='the dump dir')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

METRICS_FILE = os.path.join(args.checkpoint_dir, 'ood', args.ood_set, args.dump_dir, 'metrics.pkl')
with open(METRICS_FILE, 'rb') as handle:
    metrics = pickle.load(handle)

for key in metrics.keys():
    plt.figure()
    plt.hist(metrics[key][:, 0], alpha=0.5, label='in distribution', bins=10)
    plt.hist(metrics[key][:, 1], alpha=0.5, label='out of distribution', bins=10)
    plt.legend(loc='upper right')
    plt.title(key)
    plt.show()

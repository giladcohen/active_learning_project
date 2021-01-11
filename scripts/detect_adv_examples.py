import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import numpy as np
import json
import os
import argparse
import sys
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor
from active_learning_project.attacks.tta_ball_explorer import TTABallExplorer
from active_learning_project.utils import convert_tensor_to_image, calc_prob_wo_l, compute_roc, boolean_string

from active_learning_project.tta_utils import plot_ttas, update_useful_stats, register_intg_loss, \
    register_intg_rel_loss, register_max_rel_loss, register_rank_at_thd_rel_loss, register_rank_at_first_pred_switch, \
    register_num_pred_switches, register_mean_loss_for_initial_label, register_mean_rel_loss_for_initial_label, \
    register_intg_confidences_prime, register_intg_confidences_prime_specific, register_intg_confidences_secondary, \
    register_intg_confidences_secondary_specific, register_intg_delta_confidences_prime_rest, \
    register_intg_delta_confidences_prime_secondary_specific, register_delta_probs_prime_secondary_excl_rest

from active_learning_project.global_vars import features_index, normal_features_list, adv_features_list
import matplotlib.pyplot as plt

from art.classifiers import PyTorchClassifier
from active_learning_project.classifiers.pytorch_ext_classifier import PyTorchExtClassifier

parser = argparse.ArgumentParser(description='PyTorch adversarial robustness testing')
parser.add_argument('--src',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00/deepfool/tta_ball_rev_L2_eps_2_n_1000',
                    type=str, help='checkpoint dir')
parser.add_argument('--dest',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00/deepfool/tta_ball_rev_L2_eps_2_n_1000',
                    type=str, help='checkpoint dir')





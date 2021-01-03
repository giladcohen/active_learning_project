import pandas as pd

# features
features_index = []
normal_features_list = []
adv_features_list = []

FEATURES_RANKS = {
    'intg_loss': {'cifar10': 450},
    'intg_rel_loss': {'cifar10': 1000},
    'max_rel_loss': {'cifar10': 1000},
    'rank_at_thd_rel_loss': {'cifar10': None},  # set only THD
    'rank_at_first_pred_switch': {'cifar10': None},
    'num_pred_switches': {'cifar10': 500},
    'mean_loss_for_initial_label': {'cifar10': None},
    'mean_rel_loss_for_initial_label': {'cifar10': None},
    'intg_confidences_prime': {'cifar10': 50},
    'intg_confidences_prime_specific': {'cifar10': 350},
    'intg_confidences_secondary': {'cifar10': 50},
    'intg_confidences_secondary_specific': {'cifar10': 250},
    'intg_delta_confidences_prime_rest': {'cifar10': 50},
    'intg_delta_confidences_prime_secondary_specific': {'cifar10': 250},
    'delta_probs_prime_secondary_excl_rest': {'cifar10': None}
}

FEATURES_THD = {
    'rank_at_thd_rel_loss': {'cifar10': 0.5}
}
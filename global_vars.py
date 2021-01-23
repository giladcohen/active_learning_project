import pandas as pd

# features
features_index = []
normal_features_list = []
adv_features_list = []

# FEATURES_RANKS = {
#     'intg_loss': {'cifar10': 450},
#     'intg_rel_loss': {'cifar10': 1000},
#     'max_rel_loss': {'cifar10': 1000},
#     'rank_at_thd_rel_loss': {'cifar10': 0.5},  # set THD
#     'rank_at_first_pred_switch': {'cifar10': None},
#     'num_pred_switches': {'cifar10': 500},  # for f1:350, for f2:500
#     'mean_loss_for_initial_label': {'cifar10': None},
#     'mean_rel_loss_for_initial_label': {'cifar10': None},
#     'intg_confidences_prime': {'cifar10': 50},
#     'intg_confidences_prime_specific': {'cifar10': 350},  # for f1:450, for f2:350
#     'intg_confidences_secondary': {'cifar10': 50},
#     'intg_confidences_secondary_specific': {'cifar10': 250},
#     'intg_delta_confidences_prime_rest': {'cifar10': 50},
#     'intg_delta_confidences_prime_secondary_specific': {'cifar10': 250},  # for f1:400, for f2:250
#     'delta_probs_prime_secondary_excl_rest': {'cifar10': None}
# }

FEATURES_RANKS = {
    'intg_loss': {'cifar10': 1000},
    'intg_rel_loss': {'cifar10': 1000},
    'max_rel_loss': {'cifar10': 1000},
    'rank_at_thd_rel_loss': {'cifar10': 0.2},  # set THD
    'rank_at_first_pred_switch': {'cifar10': None},
    'num_pred_switches': {'cifar10': 900},  # for f1:350, for f2:500
    'mean_loss_for_initial_label': {'cifar10': None},
    'mean_rel_loss_for_initial_label': {'cifar10': None},
    'intg_confidences_prime': {'cifar10': 550},
    'intg_confidences_prime_specific': {'cifar10': 800},  # for f1:450, for f2:350
    'intg_confidences_secondary': {'cifar10': 100},
    'intg_confidences_secondary_specific': {'cifar10': 950},
    'intg_delta_confidences_prime_rest': {'cifar10': 450},
    'intg_delta_confidences_prime_secondary_specific': {'cifar10': 800},  # for f1:400, for f2:250
    'delta_probs_prime_secondary_excl_rest': {'cifar10': None}
}


import pandas as pd

# features
features_index = []
normal_features_list = []
adv_features_list = []

FEATURES_RANGES = [
    ['intg_loss',
     'intg_rel_loss', 'max_rel_loss', 'rank_at_thd_rel_loss', 'rank_at_first_pred_switch',
     'num_pred_switches', 'mean_loss_for_initial_label', 'mean_rel_loss_for_initial_label', 'intg_confidences_prime',
     'intg_confidences_prime_specific', 'intg_confidences_secondary', 'intg_confidences_secondary_specific',
     'intg_delta_confidences_prime_rest', 'intg_delta_confidences_prime_secondary_specific',
     'delta_probs_prime_secondary_excl_rest']
]
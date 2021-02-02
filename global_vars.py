import pandas as pd

# features
features_index = []
normal_features_list = []
adv_features_list = []

FEATURES_RANKS = {
    'intg_loss'                                        : {'cifar10' : {'regular': {'f1': 450 , 'f2': 450 }, 'robust': {'f1': 1000, 'f2': 1000}},
                                                          'cifar100': {'regular': {'f1': 350 , 'f2': 350 }, 'robust': {'f1': 1000, 'f2': 1000}},
                                                          'svhn'    : {'regular': {'f1': 800 , 'f2':   -1} , 'robust': {'f1': -1  , 'f2': -1}}
                                                          },
    'intg_rel_loss'                                    : {'cifar10' : {'regular': {'f1': 1000, 'f2': 1000}, 'robust': {'f1': 1000, 'f2': 1000}},
                                                          'cifar100': {'regular': {'f1': 1000, 'f2': 1000}, 'robust': {'f1': 1000, 'f2': 1000}},
                                                          'svhn'    : {'regular': {'f1': 1000, 'f2':   -1}, 'robust': {'f1': -1  , 'f2': -1}}
                                                          },
    'max_rel_loss'                                     : {'cifar10' : {'regular': {'f1': 1000, 'f2': 1000}, 'robust': {'f1': 1000, 'f2': 1000}},
                                                          'cifar100': {'regular': {'f1': 1000, 'f2': 1000}, 'robust': {'f1': 1000, 'f2': 1000}},
                                                          'svhn'    : {'regular': {'f1': 1000, 'f2':   -1}, 'robust': {'f1': -1, 'f2': -1}}
                                                          },
    'rank_at_thd_rel_loss'                             : {'cifar10' : {'regular': {'f1': 0.5 , 'f2': 0.5 }, 'robust': {'f1': 0.2 , 'f2': 0.2}},
                                                          'cifar100': {'regular': {'f1': 0.3 , 'f2': 0.3 }, 'robust': {'f1': 0.1 , 'f2': 0.1}},
                                                          'svhn'    : {'regular': {'f1': 0.4 , 'f2':   -1}, 'robust': {'f1': -1, 'f2': -1}}
                                                          },
    'rank_at_first_pred_switch'                        : None,
    'num_pred_switches'                                : {'cifar10' : {'regular': {'f1': 350 , 'f2': 500 }, 'robust': {'f1': 900 , 'f2': 900}},
                                                          'cifar100': {'regular': {'f1': 400 , 'f2': 400 }, 'robust': {'f1': 1000, 'f2': 1000}},
                                                          'svhn'    : {'regular': {'f1': 900 , 'f2':   -1}, 'robust': {'f1': -1, 'f2': -1}}
                                                          },
    'mean_loss_for_initial_label'                      : None,
    'mean_rel_loss_for_initial_label'                  : None,
    'intg_confidences_prime'                           : {'cifar10' : {'regular': {'f1': 50  , 'f2': 50  }, 'robust': {'f1': 550 , 'f2': 550}},
                                                          'cifar100': {'regular': {'f1': 50  , 'f2': 50  }, 'robust': {'f1': 50  , 'f2': 50}},
                                                          'svhn'    : {'regular': {'f1': 200 , 'f2': -1}, 'robust': {'f1': -1, 'f2': -1}}
                                                          },
    'intg_confidences_prime_specific'                  : {'cifar10' : {'regular': {'f1': 450 , 'f2': 350 }, 'robust': {'f1': 800 , 'f2': 800}},
                                                          'cifar100': {'regular': {'f1': 150 , 'f2': 150 }, 'robust': {'f1': 550 , 'f2': 550}},
                                                          'svhn'    : {'regular': {'f1': 800 , 'f2':   -1}, 'robust': {'f1': -1, 'f2': -1}}
                                                          },
    'intg_confidences_secondary'                       : {'cifar10' : {'regular': {'f1': 50  , 'f2': 50  }, 'robust': {'f1': 100 , 'f2': 100}},
                                                          'cifar100': {'regular': {'f1': 50  , 'f2': 50  }, 'robust': {'f1': 50  , 'f2': 50}},
                                                          'svhn'    : {'regular': {'f1': 250 , 'f2':   -1}, 'robust': {'f1': -1, 'f2': -1}}
                                                          },
    'intg_confidences_secondary_specific'              : {'cifar10' : {'regular': {'f1': 250 , 'f2': 250 }, 'robust': {'f1': 950 , 'f2': 950}},
                                                          'cifar100': {'regular': {'f1': 300 , 'f2': 300 }, 'robust': {'f1': 500 , 'f2': 500}},
                                                          'svhn'    : {'regular': {'f1': 550 , 'f2':   -1}, 'robust': {'f1': -1, 'f2': -1}}
                                                          },
    'intg_delta_confidences_prime_rest'                : {'cifar10' : {'regular': {'f1': 50  , 'f2': 50  }, 'robust': {'f1': 450 , 'f2': 450}},
                                                          'cifar100': {'regular': {'f1': 50  , 'f2': 50  }, 'robust': {'f1': 50  , 'f2': 50}},
                                                          'svhn'    : {'regular': {'f1': 200 , 'f2':   -1}, 'robust': {'f1': -1, 'f2': -1}}
                                                          },
    'intg_delta_confidences_prime_secondary_specific'  : {'cifar10' : {'regular': {'f1': 400 , 'f2': 250 }, 'robust': {'f1': 950 , 'f2': 800}},
                                                          'cifar100': {'regular': {'f1': 450 , 'f2': 450 }, 'robust': {'f1': 950 , 'f2': 950}},
                                                          'svhn'    : {'regular': {'f1': 950 , 'f2':   -1}, 'robust': {'f1': -1, 'f2': -1}}
                                                          },
    'delta_probs_prime_secondary_excl_rest'            : None
}

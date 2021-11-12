"""
Train random forest classifier using the normal images and adv images generated for:
FGSM, JSMA, PGD, Deepfool, CW, Square, and Boundary
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
import matplotlib.pyplot as plt
import scipy
import numpy as np
import json
import os
import argparse
import time
import pickle
import logging
import sys
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from active_learning_project.datasets.tta_utils import get_tta_transforms, get_tta_logits
from active_learning_project.datasets.utils import get_dataset_inds, get_ensemble_dir, get_dump_dir
from active_learning_project.utils import boolean_string, pytorch_evaluate, set_logger, get_ensemble_paths, \
    majority_vote, convert_tensor_to_image, print_Linf_dists, calc_attack_rate, get_image_shape
from active_learning_project.models.utils import get_strides, get_conv1_params, get_model
from active_learning_project.classifiers.pytorch_classifier_specific import PyTorchClassifierSpecific

parser = argparse.ArgumentParser(description='Evaluating robustness score')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--tta_input_dir', default='tta', type=str, help='The dir which holds the tta results')

parser.add_argument('--num_workers', default=20, type=int, help='Data loading threads for tta loader or random forest')

# dump
parser.add_argument('--dump_dir', default='random_forest', type=str, help='dump dir for logs and data')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

batch_size = 100

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
# CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
DUMP_DIR = os.path.join(args.checkpoint_dir, args.dump_dir)
random_forest_classifier_path = os.path.join(DUMP_DIR, 'random_forest_classifier.pkl')
os.makedirs(DUMP_DIR, exist_ok=True)
log_file = os.path.join(DUMP_DIR, 'log.log')
# dumping args to txt file
with open(os.path.join(DUMP_DIR, 'train_random_forest_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
set_logger(log_file)
logger = logging.getLogger()
rand_gen = np.random.RandomState(seed=12345)

dataset = train_args['dataset']
val_inds, _ = get_dataset_inds(dataset)
val_size = len(val_inds)

# get data:
test_loader = get_test_loader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=False)
img_shape = get_image_shape(dataset)
X_test = get_normalized_tensor(test_loader, img_shape, batch_size)
y_test = np.asarray(test_loader.dataset.targets)
classes = test_loader.dataset.classes

ATTACK_DIRS = ['fgsm_targeted', 'fgsm_targeted_eps_0.031', 'jsma_targeted', 'pgd_targeted', 'pgd_targeted_eps_0.031',
               'deepfool', 'cw_targeted', 'cw_targeted_Linf_eps_0.031', 'square']  # TODO: add boundary

# load normal tta_logits:
tta_dir = get_dump_dir(args.checkpoint_dir, args.tta_input_dir, '')
tta_logits_norm = np.load(os.path.join(tta_dir, 'tta_logits.npy'))
tta_logits_train_norm = tta_logits_norm[val_inds]

# load attacked tta_logits to train:
tta_logits_train_adv = []
for attack_dir in ATTACK_DIRS:
    tta_dir = get_dump_dir(args.checkpoint_dir, args.tta_input_dir, attack_dir)
    tta_logits_adv = np.load(os.path.join(tta_dir, 'tta_logits.npy'))
    tta_logits_train_adv.append(tta_logits_adv[val_inds])
tta_logits_train_adv = np.vstack(tta_logits_train_adv)

# reshape to features:
features_train_norm = tta_logits_train_norm.reshape((val_size, -1))
features_train_adv = tta_logits_train_adv.reshape((len(ATTACK_DIRS) * val_size, -1))

# concatenate features:
features_train = np.concatenate((features_train_norm, features_train_adv), axis=0)
labels_train = np.tile(y_test[val_inds], len(ATTACK_DIRS) + 1)

logger.info('Initializing random forest classifier for all attacks...')
clf = RandomForestClassifier(
    n_estimators=1000,
    criterion="gini",  # gini or entropy
    max_depth=None, # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
    # until all leaves contain less than min_samples_split samples.
    bootstrap=True, # Whether bootstrap samples are used when building trees.
    # If False, the whole dataset is used to build each tree.
    random_state=rand_gen,
    verbose=1000,
    n_jobs=args.num_workers
)
logger.info('Start training the classifier...')
clf.fit(features_train, labels_train)

with open(random_forest_classifier_path, "wb") as f:
    pickle.dump(clf, f)

logger.info('Successfully saved random forest classifier to {}'.format(random_forest_classifier_path))

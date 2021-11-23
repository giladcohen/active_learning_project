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
from active_learning_project.datasets.utils import get_dataset_inds, get_mini_dataset_inds, \
    get_ensemble_dir, get_dump_dir, get_boundary_val_test_inds
from active_learning_project.utils import boolean_string, pytorch_evaluate, set_logger, get_ensemble_paths, \
    majority_vote, convert_tensor_to_image, calc_attack_rate, get_image_shape
from active_learning_project.models.utils import get_strides, get_conv1_params, get_model
from active_learning_project.classifiers.pytorch_classifier_specific import PyTorchClassifierSpecific

parser = argparse.ArgumentParser(description='Evaluating robustness score')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--model', default='random_forest', type=str,
                    help='model: random_forest / logistic_regression / svm_linear / svm_rbf')
parser.add_argument('--train_exc', default='none', type=str,
                    help='attacks to exclude from RF training: none (to keep all), fgsm, jsma, pgd, deepfool, '
                         'cw, square, boundary, or all (to include only normal)')
parser.add_argument('--train_incl', default='', type=str,
                    help='attack dir to include for RF training.')

parser.add_argument('--num_workers', default=20, type=int, help='Data loading threads for tta loader or random forest')

# dump
parser.add_argument('--dump_dir', default='', type=str, help='dump dir for logs and data')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

if args.dump_dir == '':
    args.dump_dir = args.model

batch_size = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
DUMP_DIR = os.path.join(args.checkpoint_dir, args.dump_dir)
classifier_path = os.path.join(DUMP_DIR, args.model + '_classifier.pkl')
os.makedirs(DUMP_DIR, exist_ok=True)
log_file = os.path.join(DUMP_DIR, 'log.log')
# dumping args to txt file
with open(os.path.join(DUMP_DIR, 'train_' + args.model + '_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
set_logger(log_file)
logger = logging.getLogger()
rand_gen = np.random.RandomState(seed=12345)

# get data:
logger.info('==> Preparing Data..')
dataset = train_args['dataset']
val_inds, _ = get_dataset_inds(dataset)
mini_val_inds, _ = get_mini_dataset_inds(dataset)

test_loader = get_test_loader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=False)
img_shape = get_image_shape(dataset)
X_norm = get_normalized_tensor(test_loader, img_shape, batch_size)
y_gt = np.asarray(test_loader.dataset.targets)
classes = test_loader.dataset.classes

if args.train_incl != '':
    ATTACK_DIRS = [args.train_incl]
else:
    ATTACK_DIRS = ['fgsm_targeted', 'fgsm_targeted_eps_0.031', 'jsma_targeted', 'pgd_targeted', 'pgd_targeted_eps_0.031',
                   'deepfool', 'cw_targeted', 'cw_targeted_Linf_eps_0.031', 'square', 'boundary_targeted']

    if args.train_exc == 'none':
        pass
    elif args.train_exc == 'fgsm':
        ATTACK_DIRS.remove('fgsm_targeted')
        ATTACK_DIRS.remove('fgsm_targeted_eps_0.031')
    elif args.train_exc == 'jsma':
        ATTACK_DIRS.remove('jsma_targeted')
    elif args.train_exc == 'pgd':
        ATTACK_DIRS.remove('pgd_targeted')
        ATTACK_DIRS.remove('pgd_targeted_eps_0.031')
    elif args.train_exc == 'deepfool':
        ATTACK_DIRS.remove('deepfool')
    elif args.train_exc == 'cw':
        ATTACK_DIRS.remove('cw_targeted')
        ATTACK_DIRS.remove('cw_targeted_Linf_eps_0.031')
    elif args.train_exc == 'square':
        ATTACK_DIRS.remove('square')
    elif args.train_exc == 'boundary':
        ATTACK_DIRS.remove('boundary_targeted')
    elif args.train_exc == 'all':
        ATTACK_DIRS = []
    else:
        err = 'args.train_acc = {} is illegal'
        logger.error(err)
        raise AssertionError(err)

logger.info('After filtering we have: ATTACK_DIRS = {}'.format(ATTACK_DIRS))

# Model
logger.info('==> Building model..')
conv1 = get_conv1_params(dataset)
strides = get_strides(dataset)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
if 'best_net' in global_state:
    global_state = global_state['best_net']
net = get_model(train_args['net'])(num_classes=len(classes), activation=train_args['activation'], conv1=conv1, strides=strides)
net = net.to(device)
net.load_state_dict(global_state)
net.eval()  # frozen
# summary(net, (img_shape[2], img_shape[0], img_shape[1]))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

def get_val_tta_logits(attack_dir):
    is_attacked = attack_dir != ''
    is_boundary = is_attacked and 'boundary' in attack_dir
    is_cw_hacked = is_attacked and attack_dir == 'cw_targeted' and dataset == 'tiny_imagenet' and 'resnet101' in args.checkpoint_dir and 'adv_robust_vat' in args.checkpoint_dir

    tta_dir = get_dump_dir(args.checkpoint_dir, 'tta', attack_dir)
    os.makedirs(tta_dir, exist_ok=True)
    tta_file = os.path.join(tta_dir, 'tta_logits_val.npy')
    tta_file_val_test = os.path.join(tta_dir, 'tta_logits.npy')

    if not (is_boundary or is_cw_hacked):
        sel_full_inds = val_inds
        sel_mini_inds = val_inds
    else:
        sel_full_inds = mini_val_inds
        sel_mini_inds, _ = get_boundary_val_test_inds(dataset)

    if os.path.exists(tta_file):
        logger.info('loading val TTA logits from {}...'.format(tta_file))
        tta_logits = np.load(tta_file)
    elif os.path.exists(tta_file_val_test):
        logger.info('loading all TTA logits from {}...'.format(tta_file_val_test))
        tta_logits = np.load(tta_file_val_test)
        tta_logits = tta_logits[sel_mini_inds]
    else:
        logger.info('Calculating {} validation tta logits for {}. It will take couple of minutes...'
                    .format(len(sel_full_inds), attack_dir))
        tta_args = {'tta_size': 256, 'gaussian_std': 0.005, 'soft_transforms': False, 'clip_inputs': False,
                    'num_workers': args.num_workers}

        if not is_attacked:
            X = X_norm[sel_full_inds]
        else:
            X = np.load(os.path.join(args.checkpoint_dir, attack_dir, 'X_test_adv.npy'))[sel_mini_inds]
        y = y_gt[sel_full_inds]

        tta_logits = get_tta_logits(dataset, net, X, y, len(classes), tta_args)
        np.save(os.path.join(tta_dir, 'tta_logits_val.npy'), tta_logits)

    return tta_logits


train_tta_logits = []
if args.train_incl != '':
    train_tta_logits.append(get_val_tta_logits(ATTACK_DIRS[0]))
else:
    for attack_dir in [''] + ATTACK_DIRS:
        tmp_logits = get_val_tta_logits(attack_dir)
        logger.info('getting {} items from attack_dir: {}'.format(len(tmp_logits), attack_dir))
        train_tta_logits.append(tmp_logits)
train_tta_logits = np.vstack(train_tta_logits)
features_train = train_tta_logits.reshape((train_tta_logits.shape[0], -1))
assert features_train.shape[1] == len(classes) * 256

if args.train_incl != '':
    logger.info('Done collecting val tta logits. Exiting...')
    exit(0)

labels = []
for attack_dir in [''] + ATTACK_DIRS:
    if attack_dir not in ['cw_targeted', 'boundary_targeted']:
        labels.append(y_gt[val_inds])
    else:
        labels.append(y_gt[mini_val_inds])
labels_train = np.hstack(labels)

assert features_train.shape[0] == labels_train.shape[0]

logger.info('Initializing {} classifier for the selected attacks...'.format(args.model))
if args.model == 'random_forest':
    if dataset == 'tiny_imagenet':
        n_estimators = 100
    else:
        n_estimators = 1000
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",  # gini or entropy
        max_depth=None, # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
        # until all leaves contain less than min_samples_split samples.
        bootstrap=True, # Whether bootstrap samples are used when building trees.
        # If False, the whole dataset is used to build each tree.
        random_state=rand_gen,
        verbose=1000,
        n_jobs=args.num_workers
    )
elif args.model == 'logistic_regression':
    clf = LogisticRegression(multi_class='ovr', random_state=rand_gen, n_jobs=args.num_workers, verbose=1)
elif args.model == 'svm_linear':
    clf = LinearSVC(penalty='l2', loss='hinge', multi_class='ovr', random_state=rand_gen, verbose=1)
elif args.model == 'svm_rbf':
    clf = SVC(kernel='rbf', decision_function_shape='ovr', break_ties=True, random_state=rand_gen, verbose=1)
else:
    err = 'model {} is not supported'.format(args.model)
    logger.error(err)
    raise AssertionError(err)

logger.info('Start training the classifier...')
clf.fit(features_train, labels_train)

with open(classifier_path, "wb") as f:
    pickle.dump(clf, f)

logger.info('Successfully saved {} classifier to {}'.format(args.model, classifier_path))

'''Test robustness with PyTorch.'''
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

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from active_learning_project.datasets.tta_utils import get_tta_transforms, get_tta_logits
from active_learning_project.datasets.utils import get_dataset_inds, get_mini_dataset_inds, get_ensemble_dir, \
    get_dump_dir, get_boundary_val_test_inds, get_attack_inds
from active_learning_project.utils import boolean_string, pytorch_evaluate, set_logger, get_ensemble_paths, \
    majority_vote, convert_tensor_to_image, calc_attack_rate, get_image_shape, print_norm_dists, dump_imgs_to_dir
from active_learning_project.models.utils import get_strides, get_conv1_params, get_model
from active_learning_project.classifiers.pytorch_classifier_specific import PyTorchClassifierSpecific
from active_learning_project.classifiers.hybrid_classifier import HybridClassifier


parser = argparse.ArgumentParser(description='Evaluating robustness score')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--method', default='tta_counter', type=str,
                    help='simple, ensemble, tta, tta_counter, random_forest, logistic_regression, svm_linear, svm_rbf')
parser.add_argument('--attack_dir', default='boundary_targeted', type=str, help='attack directory, or None for normal images')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

# tta method params:
parser.add_argument('--tta_size', default=256, type=int, help='number of test-time augmentations')
parser.add_argument('--gaussian_std', default=0.005, type=float, help='Standard deviation of Gaussian noise')
parser.add_argument('--tta_output_dir', default='tta', type=str, help='The dir to dump the tta results for further use')
parser.add_argument('--soft_transforms', action='store_true', help='applying mellow transforms')
parser.add_argument('--clip_inputs', action='store_true', help='clipping TTA inputs between 0 and 1')
parser.add_argument('--overwrite', action='store_true', help='force calculating and saving TTA')
parser.add_argument('--num_workers', default=20, type=int, help='Data loading threads for tta loader or random forest')

# random forest params:
parser.add_argument('--classifier_dir', default='random_forest', type=str, help='The RF/LR/SVM classifier dir')

# dump
parser.add_argument('--dump_dir', default='tta_counter', type=str, help='dump dir for logs and data')
parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
is_attacked = args.attack_dir != ''
is_boundary = is_attacked and 'boundary' in args.attack_dir

if is_attacked:
    ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
    with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
        attack_args = json.load(f)
    targeted = attack_args['targeted']
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
batch_size = args.batch_size

DUMP_DIR = get_dump_dir(args.checkpoint_dir, args.dump_dir, args.attack_dir)
os.makedirs(DUMP_DIR, exist_ok=True)
log_file = os.path.join(DUMP_DIR, 'log.log')

# dumping args to txt file
with open(os.path.join(DUMP_DIR, 'eval_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

set_logger(log_file)
logger = logging.getLogger()
rand_gen = np.random.RandomState(seed=12345)

dataset = train_args['dataset']
# fetch all indices
val_inds, test_inds = get_dataset_inds(dataset)
mini_val_inds, mini_test_inds = get_mini_dataset_inds(dataset)
boundary_val_inds, boundary_test_inds = get_boundary_val_test_inds(dataset)

# get data:
test_loader = get_test_loader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=device=='cuda')
img_shape = get_image_shape(dataset)
X_test = get_normalized_tensor(test_loader, img_shape, batch_size)
y_test = np.asarray(test_loader.dataset.targets)
classes = test_loader.dataset.classes

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

# setting classifier
classifier = PyTorchClassifierSpecific(
    model=net, clip_values=(0, 1), loss=None,
    optimizer=None, input_shape=(img_shape[2], img_shape[0], img_shape[1]),
    nb_classes=len(classes), fields=['logits'])

# selecting inds automatically for an attack:
if is_attacked:
    x_inds, gt_inds = get_attack_inds(dataset,
                                      attack_args['attack'],
                                      'adv_robust_vat' in args.checkpoint_dir,
                                      'resnet101' in args.checkpoint_dir)
else:
    x_inds = None
    gt_inds = test_inds

X_test = X_test[gt_inds]
y_test = y_test[gt_inds]
y_orig_norm_preds = classifier.predict(X_test, batch_size).argmax(axis=1)
orig_norm_acc = np.mean(y_orig_norm_preds == y_test)
logger.info('Normal test accuracy: {}%'.format(100 * orig_norm_acc))

if not is_attacked:
    if args.method == 'simple':
        logger.info('done')  # already calculated above
        exit(0)
    else:
        logger.info('considering original images only...')
        X = X_test
else:
    logger.info('considering adv images of attack {}. targeted={}'
                .format(attack_args['attack'], attack_args['targeted']))
    X = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))

    if x_inds is None:
        logger.info('selecting all indices ({}) from X'.format(X.shape[0]))
    else:
        logger.info('selecing {} indices from X'.format(len(x_inds)))
        X = X[x_inds]

    print_norm_dists(X, X_test, np.inf)
    print_norm_dists(X, X_test, 2)

assert X.shape == X_test.shape, 'shape of X and X_test must be the same'
assert X.shape[0] == y_test.shape[0]

if args.method == 'simple':
    y_preds = classifier.predict(X, batch_size).argmax(axis=1)
elif args.method == 'ensemble':
    ensemble_dir = get_ensemble_dir(dataset, train_args['net'])
    networks_list = get_ensemble_paths(ensemble_dir)
    networks_list.remove(CHECKPOINT_PATH)
    num_networks = len(networks_list)
    y_preds_nets = np.nan * np.ones((X.shape[0], num_networks), dtype=np.int32)
    for j, ckpt_file in tqdm(enumerate(networks_list)):  # for network j
        logger.info('Evaluating network {}'.format(ckpt_file))
        global_state = torch.load(ckpt_file, map_location=torch.device(device))
        net.load_state_dict(global_state['best_net'])
        y_preds_nets[:, j] = classifier.predict(X, batch_size=batch_size).argmax(axis=1)
    assert not np.isnan(y_preds_nets).any()
    y_preds_nets = y_preds_nets.astype(np.int32)
    y_preds = np.apply_along_axis(majority_vote, axis=1, arr=y_preds_nets)
elif args.method in ['tta', 'tta_counter']:
    tta_dir = get_dump_dir(args.checkpoint_dir, args.tta_output_dir, args.attack_dir)
    os.makedirs(tta_dir, exist_ok=True)
    tta_file = os.path.join(tta_dir, 'tta_logits_test.npy')
    tta_file_val_test = os.path.join(tta_dir, 'tta_logits.npy')
    if args.overwrite:
        logger.info('Calculating tta logits (overwrite). It will take couple of minutes...')
        tta_logits = get_tta_logits(dataset, net, X, y_test, len(classes), args.__dict__)
        np.save(os.path.join(tta_dir, 'tta_logits_test.npy'), tta_logits)
    else:
        if os.path.exists(tta_file):
            logger.info('loading test TTA logits from {}...'.format(tta_file))
            tta_logits = np.load(tta_file)
        elif os.path.exists(tta_file_val_test):
            logger.info('loading all TTA logits from {}...'.format(tta_file_val_test))
            tta_logits = np.load(tta_file)[test_inds]
        else:
            logger.info('Did not find tta_logits in:\n{}\n{}'.format(tta_file, tta_file_val_test))
            logger.info('Calculating test tta logits. It will take couple of minutes...')
            tta_logits = get_tta_logits(dataset, net, X, y_test, len(classes), args.__dict__)
            np.save(os.path.join(tta_dir, 'tta_logits_test.npy'), tta_logits)

    if args.method == 'tta':
        y_preds = tta_logits.sum(axis=1).argmax(axis=1)
    elif args.method == 'tta_counter':
        tta_probs = scipy.special.softmax(tta_logits, axis=2)
        tta_preds = tta_probs.argmax(axis=2)
        y_preds = np.apply_along_axis(majority_vote, axis=1, arr=tta_preds)
    else:
        logger.error('How did I get here?')
elif args.method in ['random_forest', 'logistic_regression', 'svm_linear', 'svm_rbf']:
    model_dir = os.path.join(args.checkpoint_dir, args.classifier_dir)
    model_path = os.path.join(model_dir, args.method + '_classifier.pkl')
    tta_dir = get_dump_dir(args.checkpoint_dir, args.tta_output_dir, args.attack_dir)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if args.method in ['random_forest', 'logistic_regression']:
        model.n_jobs = max(1, args.num_workers)  # overwrite
        probability = True
    else:
        probability = False
    model.verbose = 0
    hybrid_classifier = HybridClassifier(
        dnn_model=net,
        rf_model=model,
        dataset=dataset,
        tta_args=args.__dict__,
        input_shape=(img_shape[2], img_shape[0], img_shape[1]),
        nb_classes=len(classes),
        clip_values=(0, 1),
        fields=['logits'],
        tta_dir=tta_dir,
        probability=probability
    )
    y_preds = hybrid_classifier.predict(X, batch_size)
    if probability:
        y_preds = y_preds.argmax(axis=1)
else:
    raise AssertionError('unknown method {}'.format(args.method))

# metrics calculation:
acc = np.mean(y_preds == y_test)
logger.info('Test accuracy: {}%'.format(100.0 * acc))

logger.handlers[0].flush()
if not is_attacked:
    logger.info('Done')
    exit(0)

attack_rate, f2_inds = calc_attack_rate(y_preds, y_orig_norm_preds, y_test)
np.save(os.path.join(DUMP_DIR, 'f2_inds.npy'), f2_inds)
logger.info('attack success rate: {}%'.format(100.0 * attack_rate))
logger.info('After considering only successful attacks on correct DNN predictions, we have these distances:')
print_norm_dists(X[f2_inds], X_test[f2_inds], np.inf)
print_norm_dists(X[f2_inds], X_test[f2_inds], 2)

plot_dir = os.path.join(ATTACK_DIR, 'images')
if not os.path.exists(plot_dir):
    dump_imgs_to_dir(X, plot_dir)

logger.handlers[0].flush()

exit(0)
# debug:
# clipping
x_clipped = torch.clip(x, 0.0, 1.0)
#x_img = convert_tensor_to_image(x_clipped.detach().cpu().numpy())
x_img = convert_tensor_to_image(X_test)
x_adv = convert_tensor_to_image(X)
for i in range(240, 245):
    plt.imshow(x_img[i])
    plt.show()
    plt.imshow(x_adv[i])
    plt.show()

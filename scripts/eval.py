'''Test CIFAR10 robustness with PyTorch.'''
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
from active_learning_project.datasets.tta_dataset import TTADataset
from active_learning_project.datasets.tta_transforms import get_tta_transforms
from active_learning_project.datasets.utils import get_mini_dataset_inds, get_ensemble_dir, get_dump_dir
from active_learning_project.utils import boolean_string, pytorch_evaluate, set_logger, get_model, get_ensemble_paths, \
    majority_vote, convert_tensor_to_image, print_Linf_dists
from art.classifiers import PyTorchClassifier

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/adv_robust_trades', type=str, help='checkpoint dir')
parser.add_argument('--method', default='simple', type=str, help='simple, ensemble, tta, random_forest')
parser.add_argument('--attack_dir', default='cw_targeted', type=str, help='attack directory, or None for normal images')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='Data loading threads')

# tta method params:
parser.add_argument('--tta_size', default=1024, type=int, help='number of test-time augmentations')
parser.add_argument('--gaussian_std', default=0.005, type=float, help='Standard deviation of Gaussian noise')  # was 0.0125
parser.add_argument('--tta_output_dir', default='tta_debug', type=str, help='The dir to dump the tta results for further use')
parser.add_argument('--soft_transforms', action='store_true', help='applying mellow transforms')
parser.add_argument('--clip_inputs', action='store_true', help='clipping TTA inputs between 0 and 1')

# dump
parser.add_argument('--dump_dir', default='debug', type=str, help='dump dir for logs and data')
parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
if args.attack_dir != '':
    ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
    with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
        attack_args = json.load(f)
    targeted = attack_args['targeted']
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
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
val_inds, test_inds = get_mini_dataset_inds(dataset)
test_size = len(test_inds)

# get data:
test_loader = get_test_loader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=1,
    pin_memory=device=='cuda')
X_test = get_normalized_tensor(test_loader, batch_size)
y_test = np.asarray(test_loader.dataset.targets)
classes = test_loader.dataset.classes

# get network:
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
net = get_model(train_args['net'])(num_classes=len(classes), activation=train_args['activation'])
net = net.to(device)
net.load_state_dict(global_state['best_net'])
net.eval()  # frozen
# summary(net, (3, 32, 32))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# setting classifier
classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=None,
                               optimizer=None, input_shape=(3, 32, 32), nb_classes=len(classes))

y_gt = y_test[test_inds]
y_orig_norm_preds = classifier.predict(X_test[test_inds], batch_size).argmax(axis=1)
orig_norm_acc = np.mean(y_orig_norm_preds == y_gt)
logger.info('Normal test accuracy: {}%'.format(100 * orig_norm_acc))

if args.attack_dir == '':
    if args.method == 'simple':
        logger.info('done')  # already calculated above
        exit(0)
    else:
        logger.info('considering original images only...')
        X = X_test
else:
    logger.info('considering adv images of attack {}. targeted={}'.format(attack_args['attack'], attack_args['targeted']))
    X = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
    y_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy')) if attack_args['targeted'] else None
    print_Linf_dists(X[test_inds], X_test[test_inds])

if args.method == 'simple':
    y_preds = classifier.predict(X[test_inds], batch_size).argmax(axis=1)
elif args.method == 'ensemble':
    ensemble_dir = get_ensemble_dir(dataset, train_args['net'])
    networks_list = get_ensemble_paths(ensemble_dir)
    networks_list.remove(CHECKPOINT_PATH)
    num_networks = len(networks_list)
    y_preds_nets = np.nan * np.ones((test_size, num_networks), dtype=np.int32)
    for j, ckpt_file in tqdm(enumerate(networks_list)):  # for network j
        logger.info('Evaluating network {}'.format(ckpt_file))
        global_state = torch.load(ckpt_file, map_location=torch.device(device))
        net.load_state_dict(global_state['best_net'])
        y_preds_nets[:, j] = classifier.predict(X[test_inds], batch_size=batch_size).argmax(axis=1)
    assert not np.isnan(y_preds_nets).any()
    y_preds_nets = y_preds_nets.astype(np.int32)
    y_preds = np.apply_along_axis(majority_vote, axis=1, arr=y_preds_nets)
elif args.method == 'tta':
    tta_dir = get_dump_dir(args.checkpoint_dir, args.tta_output_dir, args.attack_dir)
    os.makedirs(tta_dir, exist_ok=True)
    tta_file = os.path.join(tta_dir, 'tta_logits.npy')
    if os.path.exists(tta_file):
        logger.info('tta_logits exists in {}. Loading it.'.format(tta_file))
        tta_logits = np.load(tta_file)
    else:
        logger.info('Calculating tta_logits.npy. (It might take a while...)')
        tta_transforms = get_tta_transforms(dataset, args.gaussian_std, args.soft_transforms, args.clip_inputs)
        tta_dataset = TTADataset(
            torch.from_numpy(X),
            torch.from_numpy(y_test),
            args.tta_size,
            transform=tta_transforms)
        tta_loader = torch.utils.data.DataLoader(
            tta_dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=device=='cuda')

        tta_logits = np.nan * np.ones((X.shape[0], args.tta_size, len(classes)), dtype=np.float32)
        cnt = 0
        with torch.no_grad():
            for batch_idx, (x, y) in tqdm(enumerate(tta_loader)):
                x, y = x.reshape((-1,) + tta_dataset.img_shape), y.reshape(-1)
                x, y = x.to(device), y.to(device)
                b = cnt
                e = b + y.size(0)
                tta_logits[b:e] = net(x)['logits'].cpu().numpy().reshape(-1, args.tta_size, len(classes))
                cnt += y.size(0)
        assert cnt == X.shape[0]
        assert not np.isnan(tta_logits).any()
        logger.info('Dumping TTA logits to {}'.format(tta_dir))
        np.save(os.path.join(tta_dir, 'tta_logits.npy'), tta_logits)

    # testing only test_inds:
    tta_logits = tta_logits[test_inds]
    # tta_probs = scipy.special.softmax(tta_logits, axis=2)
    # tta_preds = tta_probs.argmax(axis=2)
    # y_preds = np.apply_along_axis(majority_vote, axis=1, arr=tta_preds)
    y_preds = tta_logits.sum(axis=1).argmax(axis=1)

# metrics calculation:
acc = np.mean(y_preds == y_gt)
logger.info('Test accuracy: {}%'.format(100 * acc))

exit(0)
# debug:
x_img = convert_tensor_to_image(x.detach().cpu().numpy())
for i in range(0, 5):
    plt.imshow(x_img[i])
    plt.show()

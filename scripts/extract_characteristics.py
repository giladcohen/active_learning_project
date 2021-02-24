import matplotlib
import platform
# Force matplotlib to not use any Xwindows backend.

if platform.system() == 'Linux':
    matplotlib.use('Agg')

import logging
import numpy as np
import tensorflow as tf
import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import sys

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import sklearn.covariance
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.datasets.train_val_test_data_loaders import get_normalized_tensor, get_test_loader, \
    get_loader_with_specific_inds
from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.classifiers.pytorch_ext_classifier import PyTorchExtClassifier
from lid_adversarial_subspace_detection.util import mle_batch

STDEVS = {'cifar10' : {'fgsm': 0.01, 'jsma': 0.0534, 'pgd': 0.0089, 'deepfool': 0.0101, 'cw': 0.0148, 'ead': 0.0085},
          'cifar100': {'fgsm': 0.01, 'jsma': 0.0560, 'pgd': 0.0092, 'deepfool': 0.0029, 'cw': 0.0233, 'ead': 0.0059},
          'svhn'    : {'fgsm': 0.01, 'jsma': 0.0467, 'pgd': 0.0093, 'deepfool': 0.0227, 'cw': 0.0186, 'ead': 0.0133}
          }

parser = argparse.ArgumentParser(description='Feature extraction of LID, Mahalanobis, and ours')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='cw_targeted', type=str, help='attack directory')
parser.add_argument('--defense', default='mahalanobis', type=str, help='reverse dir')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

# for LID:
parser.add_argument('--k_nearest', default=-1, type=int, help='number of nearest neighbors to use for LID/DkNN detection')

# for mahalanobis:
parser.add_argument('--magnitude', default=-1, type=float, help='number of nearest neighbors to use for LID/DkNN detection')

# for ours:
parser.add_argument('--norm', default='L2', type=str, help='norm or ball distance')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
args.dataset = train_args['dataset']
args.net_type = 'robust' if 'adv_robust/' in args.checkpoint_dir else 'regular'

CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

NORMAL_SAVE_DIR = os.path.join(args.checkpoint_dir, 'normal')
SAVE_DIR = os.path.join(ATTACK_DIR, args.defense)  # will be changed by params
os.makedirs(os.path.join(NORMAL_SAVE_DIR), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR), exist_ok=True)

batch_size = args.batch_size
rand_gen = np.random.RandomState(seed=12345)

# Data
print('==> Preparing data..')
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
train_inds = np.asarray(global_state['train_inds'])
val_inds   = np.asarray(global_state['val_inds'])

trainloader = get_loader_with_specific_inds(
    dataset=train_args['dataset'],
    batch_size=batch_size,
    is_training=False,
    indices=train_inds,
    num_workers=1,
    pin_memory=True
)

testloader = get_test_loader(
    dataset=train_args['dataset'],
    batch_size=batch_size,
    num_workers=1,
    pin_memory=True
)

classes = testloader.dataset.classes
test_size = len(testloader.dataset)
test_inds = np.arange(test_size)

X_test           = get_normalized_tensor(testloader, batch_size)
y_test           = np.asarray(testloader.dataset.targets)
X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
if targeted:
    y_test_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))

np.save(os.path.join(args.checkpoint_dir, 'normal', 'y_test.npy'), y_test)

# Model
print('==> Building model..')
if train_args['net'] == 'resnet34':
    net = ResNet34(num_classes=len(classes), activation=train_args['activation'])
elif train_args['net'] == 'resnet101':
    net = ResNet101(num_classes=len(classes), activation=train_args['activation'])
else:
    raise AssertionError("network {} is unknown".format(train_args['net']))
net = net.to(device)

# summary(net, (3, 32, 32))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net.load_state_dict(global_state['best_net'])

criterion = nn.CrossEntropyLoss()
criterion_unreduced = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(
    net.parameters(),
    lr=train_args['lr'],
    momentum=train_args['mom'],
    weight_decay=0.0,  # train_args['wd'],
    nesterov=train_args['mom'] > 0)

# get and assert preds:
net.eval()
classifier = PyTorchExtClassifier(model=net, clip_values=(0, 1), loss=criterion, loss2=criterion_unreduced,
                                  optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(classes))

y_test_logits = classifier.predict(X_test, batch_size=batch_size)
y_test_preds = y_test_logits.argmax(axis=1)
try:
    if os.path.exists(os.path.join(os.path.join(ATTACK_DIR, 'y_test_logits.npy'))):
        np.testing.assert_array_almost_equal_nulp(y_test_logits, np.load(os.path.join(ATTACK_DIR, 'y_test_logits.npy')))
    if os.path.exists(os.path.join(os.path.join(ATTACK_DIR, 'y_test_preds.npy'))):
        np.testing.assert_array_almost_equal_nulp(y_test_preds, np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy')))
except AssertionError as e:
    raise AssertionError('{}\nAssert failed for y_test for ATTACK_DIR={}'.format(e, ATTACK_DIR))
np.save(os.path.join(args.checkpoint_dir, 'y_test_logits.npy'), y_test_logits)
np.save(os.path.join(args.checkpoint_dir, 'y_test_preds.npy'), y_test_preds)

y_test_adv_logits = classifier.predict(X_test_adv, batch_size=batch_size)
y_test_adv_preds = y_test_adv_logits.argmax(axis=1)
try:
    if os.path.exists(os.path.join(os.path.join(ATTACK_DIR, 'y_test_adv_logits.npy'))):
        np.testing.assert_array_almost_equal_nulp(y_test_adv_logits, np.load(os.path.join(ATTACK_DIR, 'y_test_adv_logits.npy')))
    if os.path.exists(os.path.join(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))):
        np.testing.assert_array_almost_equal_nulp(y_test_adv_preds, np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy')))
except AssertionError as e:
    raise AssertionError('{}\nAssert failed for y_test for ATTACK_DIR={}'.format(e, ATTACK_DIR))
np.save(os.path.join(args.checkpoint_dir, 'y_test_adv_logits.npy'), y_test_adv_logits)
np.save(os.path.join(args.checkpoint_dir, 'y_test_adv_preds.npy'), y_test_adv_preds)

# what are the samples we care about? net_succ (not attack_succ. it is irrelevant)
val_inds     = np.load(os.path.join(ATTACK_DIR, 'inds', 'val_inds.npy'))
f0_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f0_inds_val.npy'))
f1_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f1_inds_val.npy'))
f2_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f2_inds_val.npy'))
f3_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f3_inds_val.npy'))
test_inds    = np.load(os.path.join(ATTACK_DIR, 'inds', 'test_inds.npy'))
f0_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f0_inds_test.npy'))
f1_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f1_inds_test.npy'))
f2_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f2_inds_test.npy'))
f3_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f3_inds_test.npy'))

X_test_val     = X_test[val_inds]
X_test_val_adv = X_test_adv[val_inds]

X_test_test     = X_test[test_inds]
X_test_test_adv = X_test_adv[test_inds]

# get noisy images
def get_noisy_samples(X, std):
    """ Add Gaussian noise to the samples """
    # std = STDEVS[subset][FLAGS.dataset][FLAGS.attack]
    X_noisy = np.clip(X + rand_gen.normal(loc=0.0, scale=std, size=X.shape), 0, 1)
    return X_noisy

# DEBUG: testing different scale so that L2 perturbation is the same
# diff_adv    = X_test_val_adv.reshape((len(X_test_val), -1)) - X_test_val.reshape((len(X_test_val), -1))
# l2_diff_adv = np.linalg.norm(diff_adv, axis=1).mean()
# for std in np.arange(0.013, 0.014, 0.0001):
#     X_test_val_noisy = get_noisy_samples(X_test_val, std)
#     diff = X_test_val_noisy.reshape((len(X_test_val), -1)) - X_test_val.reshape((len(X_test_val), -1))
#     l2_diff = np.linalg.norm(diff, axis=1).mean()
#     print('for std={}: diff of L2 perturbations is {}'.format(std, l2_diff - l2_diff_adv))

# diff_adv    = X_test_test_adv.reshape((len(X_test_test), -1)) - X_test_test.reshape((len(X_test_test), -1))
# l2_diff_adv = np.linalg.norm(diff_adv, axis=1).mean()
# for std in np.arange(0.003, 0.004, 0.0001):
#     X_test_test_noisy = get_noisy_samples(X_test_test, std)
#     diff = X_test_test_noisy.reshape((len(X_test_test), -1)) - X_test_test.reshape((len(X_test_test), -1))
#     l2_diff = np.linalg.norm(diff, axis=1).mean()
#     print('for std={}: diff of L2 perturbations is {}'.format(std, l2_diff - l2_diff_adv))

noisy_file = os.path.join(ATTACK_DIR, 'X_test_val_noisy.npy')
if os.path.isfile(noisy_file):
    print('Loading test_val noisy samples from {}'.format(noisy_file))
    X_test_val_noisy = np.load(noisy_file).astype(np.float32)
else:
    print('Crafting test_val noisy samples for {}.'.format(ATTACK_DIR))
    X_test_val_noisy = get_noisy_samples(X_test_val, std=STDEVS[train_args['dataset']][attack_args['attack']]).astype(np.float32)
    np.save(noisy_file, X_test_val_noisy)

noisy_file = os.path.join(ATTACK_DIR, 'X_test_test_noisy.npy')
if os.path.isfile(noisy_file):
    print('Loading test_test noisy samples from {}'.format(noisy_file))
    X_test_test_noisy = np.load(noisy_file).astype(np.float32)
else:
    print('Crafting test_test noisy samples for {}.'.format(ATTACK_DIR))
    X_test_test_noisy = get_noisy_samples(X_test_test, std=STDEVS[train_args['dataset']][attack_args['attack']]).astype(np.float32)
    np.save(noisy_file, X_test_test_noisy)

# print stats for test_val
for s_type, subset in zip(['normal', 'noisy', 'adversarial'], [X_test_val, X_test_val_noisy, X_test_val_adv]):
    # acc = model_eval(sess, x, y, logits, subset, y_val, args=eval_params)
    # print("Model accuracy on the %s val set: %0.2f%%" % (s_type, 100 * acc))
    # Compute and display average perturbation sizes
    if not s_type == 'normal':
        # print for test:
        diff    = subset.reshape((len(subset), -1)) - X_test_val.reshape((len(subset), -1))
        l2_diff = np.linalg.norm(diff, axis=1).mean()
        print("Average L-2 perturbation size of the %s test_val set: %0.4f" % (s_type, l2_diff))

# print stats for test_test
for s_type, subset in zip(['normal', 'noisy', 'adversarial'], [X_test_test, X_test_test_noisy, X_test_test_adv]):
    # acc = model_eval(sess, x, y, logits, subset, y_test, args=eval_params)
    # print("Model accuracy on the %s test set: %0.2f%%" % (s_type, 100 * acc))
    # Compute and display average perturbation sizes
    if not s_type == 'normal':
        # print for test:
        diff    = subset.reshape((len(subset), -1)) - X_test_test.reshape((len(subset), -1))
        l2_diff = np.linalg.norm(diff, axis=1).mean()
        print("Average L-2 perturbation size of the %s test_test set: %0.4f" % (s_type, l2_diff))

X_test_val      = X_test[f1_inds_val]
X_test_val_adv  = X_test_adv[f1_inds_val]
X_test_test     = X_test[f1_inds_test]
X_test_test_adv = X_test_adv[f1_inds_test]

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("X_pos: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("X_neg: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def get_lids_random_batch(X, X_noisy, X_adv, k=args.k_nearest, batch_size=100):
    """
    :param X: normal images
    :param X_noisy: noisy images
    :param X_adv: advserial images
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
             lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """

    lid_dim = 1
    print("Number of layers to estimate: ", lid_dim)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start

        lid_batch       = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_adv   = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_noisy = np.zeros(shape=(n_feed, lid_dim))

        X_act       = net(torch.tensor(X[start:end]      , device=device))['logits'].cpu().detach()
        X_act       = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))

        X_adv_act   = net(torch.tensor(X_adv[start:end]  , device=device))['logits'].cpu().detach()
        X_adv_act   = np.asarray(X_adv_act, dtype=np.float32).reshape((n_feed, -1))

        X_noisy_act = net(torch.tensor(X_noisy[start:end], device=device))['logits'].cpu().detach()
        X_noisy_act = np.asarray(X_noisy_act, dtype=np.float32).reshape((n_feed, -1))

        lid_batch[:, 0]       = mle_batch(X_act, X_act, k=k)
        lid_batch_adv[:, 0]   = mle_batch(X_act, X_adv_act, k=k)
        lid_batch_noisy[:, 0] = mle_batch(X_act, X_noisy_act, k=k)

        return lid_batch, lid_batch_noisy, lid_batch_adv

    lids = []
    lids_adv = []
    lids_noisy = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_noisy, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)
        lids_noisy.extend(lid_batch_noisy)

    lids       = np.asarray(lids, dtype=np.float32)
    lids_noisy = np.asarray(lids_noisy, dtype=np.float32)
    lids_adv   = np.asarray(lids_adv, dtype=np.float32)

    return lids, lids_noisy, lids_adv

def get_lid(X, X_noisy, X_adv, k, batch_size=100):
    print('Extract local intrinsic dimensionality: k = %s' % k)
    lids_normal, lids_noisy, lids_adv = get_lids_random_batch(X, X_noisy, X_adv, k, batch_size)
    print("lids_normal:", lids_normal.shape)
    print("lids_noisy:", lids_noisy.shape)
    print("lids_adv:", lids_adv.shape)

    lids_pos = lids_adv
    lids_neg = np.concatenate((lids_normal, lids_noisy))
    artifacts, labels = merge_and_generate_labels(lids_pos, lids_neg)

    return artifacts, labels

def sample_estimator(net, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    net.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, volatile=True)
        output, out_features = net.feature_list(data)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision

start = time.time()
if args.defense == 'lid':
    if args.k_nearest == -1:
        k_vec = np.arange(10, 41, 2)
    else:
        k_vec = [args.k_nearest]

    for k in tqdm(k_vec):
        print('Extracting LID characteristics for k={}'.format(k))

        # for test-val set
        characteristics, label = get_lid(X_test_val, X_test_val_noisy, X_test_val_adv, k, 100)
        print("LID train: [characteristic shape: ", characteristics.shape, ", label shape: ", label.shape)
        file_name = os.path.join(SAVE_DIR, 'k_{}_train.npy'.format(k))
        data = np.concatenate((characteristics, label), axis=1)
        np.save(file_name, data)
        end_val = time.time()
        print('total feature extraction time for val: {} sec'.format(end_val - start))

        # for test-test set
        characteristics, label = get_lid(X_test_test, X_test_test_noisy, X_test_test_adv, k, 100)
        print("LID test: [characteristic shape: ", characteristics.shape, ", label shape: ", label.shape)
        file_name = os.path.join(SAVE_DIR, 'k_{}_test.npy'.format(k))
        data = np.concatenate((characteristics, label), axis=1)
        np.save(file_name, data)
        end_test = time.time()
        print('total feature extraction time for test: {} sec'.format(end_test - end_val))

def sample_estimator(model, num_classes, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    model.eval()
    with torch.no_grad():
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct, total = 0, 0
        num_output = 1
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(num_output):
            temp_list = []
            for j in range(num_classes):
                temp_list.append(0)
            list_features.append(temp_list)

        for data, target in train_loader:
            total += data.size(0)
            data = data.cuda()
            data = Variable(data)
            output = model(data)['logits']
            # output, out_features = out['logits'], out['logits']

            out_features = output.view(output.size(0), output.size(1), -1)
            out_features = torch.mean(out_features, dim=2)

            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()

            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                out_count = 0
                if num_sample_per_class[label] == 0:
                    list_features[out_count][label] = out_features[i].view(1, -1)
                else:
                    list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out_features[i].view(1, -1)), 0)
                num_sample_per_class[label] += 1

        sample_class_mean = []
        out_count = 0
        num_feature = num_classes
        temp_list = torch.Tensor(num_classes, num_feature).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], dim=0)
        sample_class_mean.append(temp_list)

        precision = []
        for k in range(num_output):
            X = 0
            for i in range(num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), dim=0)

            # find inverse
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().cuda()
            precision.append(temp_precision)

        print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

        return sample_class_mean, precision

def get_Mahalanobis_score_adv(model, X, num_classes, magnitude, sample_mean, precision):
    model.eval()
    Mahalanobis = []
    layer_index = 0
    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    total = X_tensor.size(0)

    num_batches = int(np.ceil(total / batch_size))
    for i_batch in range(num_batches):
        with torch.enable_grad():
            start = i_batch * batch_size
            end = np.minimum(total, (i_batch + 1) * batch_size)
            n_feed = end - start
            data = X_tensor[start:end]
            data = Variable(data, requires_grad=True)

            out_features = model(data)['logits']
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)

            gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = out_features - batch_sample_mean
                term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    gaussian_score = term_gau.view(-1,1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

            # Input_processing
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
            zero_f = out_features - Variable(batch_sample_mean)
            pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
            loss = torch.mean(-pure_gau)

            # zero all grads in model:
            classifier._model.zero_grad()
            if data.grad is not None:
                data.grad.zero_()

            loss.backward()
            gradient = torch.ge(data.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            # scale hyper params given from the official deep_Mahalanobis_detector repo:
            RED_SCALE = 0.2023
            GREEN_SCALE = 0.1994
            BLUE_SCALE = 0.2010

            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / RED_SCALE)
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / GREEN_SCALE)
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / BLUE_SCALE)

            tempInputs = torch.add(input=data.data, other=gradient, alpha=-magnitude)

        with torch.no_grad():
            noise_out_features = model(Variable(tempInputs))['logits']
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
            noise_gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = noise_out_features.data - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    noise_gaussian_score = term_gau.view(-1, 1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)

        Mahalanobis.extend(noise_gaussian_score.cpu().detach().numpy())

    return Mahalanobis

def get_mahalanobis(X, X_noisy, X_adv, magnitude, sample_mean, precision, set):
    first_pass = True
    # for layer in model.net.keys():

    layer = 'logits'
    print('Calculating Mahalanobis characteristics for set {}, {}'.format(set, layer))

    # val
    M_in = get_Mahalanobis_score_adv(net, X, len(classes), magnitude, sample_mean, precision)
    M_in = np.asarray(M_in, dtype=np.float32)

    M_out = get_Mahalanobis_score_adv(net, X_adv, len(classes), magnitude, sample_mean, precision)
    M_out = np.asarray(M_out, dtype=np.float32)

    M_noisy = get_Mahalanobis_score_adv(net, X_noisy, len(classes), magnitude, sample_mean, precision)
    M_noisy = np.asarray(M_noisy, dtype=np.float32)

    if first_pass:
        Mahalanobis_in    = M_in.reshape((M_in.shape[0], -1))
        Mahalanobis_out   = M_out.reshape((M_out.shape[0], -1))
        Mahalanobis_noisy = M_noisy.reshape((M_noisy.shape[0], -1))
        first_pass = False
    else:
        Mahalanobis_in    = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
        Mahalanobis_out   = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
        Mahalanobis_noisy = np.concatenate((Mahalanobis_noisy, M_noisy.reshape((M_noisy.shape[0], -1))), axis=1)

    Mahalanobis_neg = np.concatenate((Mahalanobis_in, Mahalanobis_noisy))
    Mahalanobis_pos = Mahalanobis_out
    characteristics, labels = merge_and_generate_labels(Mahalanobis_pos, Mahalanobis_neg)

    return characteristics, labels

if args.defense == 'mahalanobis':
    print('get sample mean and covariance of the training set...')

    sample_mean, precision = sample_estimator(net, len(classes), trainloader)
    print('Done calculating: sample_mean, precision.')

    if args.magnitude == -1:
        magnitude_vec = np.array([0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01])
    else:
        magnitude_vec = [args.magnitude]

    for magnitude in tqdm(magnitude_vec):
        print('Extracting Mahalanobis characteristics for magnitude={}'.format(magnitude))

        # val
        characteristics, label = get_mahalanobis(X_test_val, X_test_val_noisy, X_test_val_adv, magnitude, sample_mean, precision, 'train')
        print("Mahalanobis train: [characteristic shape: ", characteristics.shape, ", label shape: ", label.shape)
        file_name = os.path.join(SAVE_DIR, 'mag_{}_train.npy'.format(magnitude))
        data = np.concatenate((characteristics, label), axis=1)
        np.save(file_name, data)
        end_val = time.time()
        print('total feature extraction time for val: {} sec'.format(end_val - start))

        # test
        characteristics, label = get_mahalanobis(X_test_test, X_test_test_noisy, X_test_test_adv, magnitude, sample_mean, precision, 'test')
        print("Mahalanobis test: [characteristic shape: ", characteristics.shape, ", label shape: ", label.shape)
        file_name = os.path.join(SAVE_DIR, 'mag_{}_test.npy'.format(magnitude))
        data = np.concatenate((characteristics, label), axis=1)
        np.save(file_name, data)
        end_test = time.time()
        print('total feature extraction time for test: {} sec'.format(end_test - end_val))

# elif args.defense == 'ours':
#     if args.norm == 'L_inf':
#         norm = np.inf
#     elif args.norm == 'L1':
#         norm = 1
#     elif args.norm == 'L2':
#         norm = 2
#     else:
#         raise AssertionError('norm {} is not acceptible'.format(args.norm))

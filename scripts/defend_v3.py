import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary

from tqdm import tqdm
import numpy as np
import json
import os
import argparse
import sys
import scipy
from datetime import datetime
from sklearn.decomposition import PCA

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from active_learning_project.attacks.ball_explorer import BallExplorer
from active_learning_project.utils import convert_tensor_to_image, boolean_string

import matplotlib.pyplot as plt

from art.classifiers import PyTorchClassifier
from active_learning_project.classifiers.pytorch_ext_classifier import PyTorchExtClassifier


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='deepfool', type=str, help='attack directory')
parser.add_argument('--save_dir', default='debug', type=str, help='reverse dir')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--subset', default=500, type=int, help='attack only subset of test set')

# for exploration
parser.add_argument('--norm', default='L2', type=str, help='norm or ball distance')
parser.add_argument('--eps', default=4.0, type=float, help='the ball radius for exploration')
parser.add_argument('--num_points', default=300, type=int, help='the number of gradients to sample')
parser.add_argument('--wlg', default=False, type=boolean_string, help='include losses gradients')
parser.add_argument('--wpg', default=False, type=boolean_string, help='include preds attack')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)

CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

assert args.save_dir != ''
SAVE_DIR = os.path.join(ATTACK_DIR, args.save_dir)
# saving current args:
os.makedirs(SAVE_DIR, exist_ok=True)
with open(os.path.join(SAVE_DIR, 'run_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

batch_size = args.batch_size

rand_gen = np.random.RandomState(seed=12345)

# Data
print('==> Preparing data..')
testloader = get_test_loader(
    dataset=train_args['dataset'],
    batch_size=batch_size,
    num_workers=1,
    pin_memory=True
)

global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
train_inds = np.asarray(global_state['train_inds'])
val_inds = np.asarray(global_state['val_inds'])
classes = testloader.dataset.classes
test_size = len(testloader.dataset)
test_inds = np.arange(test_size)

X_test           = get_normalized_tensor(testloader, batch_size)
y_test           = np.asarray(testloader.dataset.targets)

X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
if targeted:
    y_test_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))

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
    np.testing.assert_array_almost_equal_nulp(y_test_preds, np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy')))
    np.save(os.path.join(ATTACK_DIR, 'y_test_logits.npy'), y_test_logits)
except AssertionError as e:
    raise AssertionError('{}\nAssert failed for y_test_preds for ATTACK_DIR={}'.format(e, ATTACK_DIR))

y_test_adv_logits = classifier.predict(X_test_adv, batch_size=batch_size)
y_test_adv_preds = y_test_adv_logits.argmax(axis=1)
try:
    np.testing.assert_array_almost_equal_nulp(y_test_adv_preds, np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy')))
    np.save(os.path.join(ATTACK_DIR, 'y_test_adv_logits.npy'), y_test_adv_logits)
except AssertionError as e:
    raise AssertionError('{}\nAssert failed for y_test_adv_logits for ATTACK_DIR={}'.format(e, ATTACK_DIR))

subset = attack_args.get('subset', -1)  # default to attack
# may be overriden:
if args.subset != -1:
    subset = args.subset

if subset != -1:  # if debug run
    X_test = X_test[:subset]
    y_test = y_test[:subset]
    X_test_adv = X_test_adv[:subset]
    if targeted:
        y_test_adv = y_test_adv[:subset]
    y_test_logits = y_test_logits[:subset]
    y_test_preds = y_test_preds[:subset]
    y_test_adv_logits = y_test_adv_logits[:subset]
    y_test_adv_preds = y_test_adv_preds[:subset]
    test_size = subset

# what are the samples we care about? net_succ (not attack_succ. it is irrelevant)
f0_inds = []  # net_fail
f1_inds = []  # net_succ
f2_inds = []  # net_succ AND attack_flip
f3_inds = []  # net_succ AND attack_flip AND attack_succ

for i in range(test_size):
    f1 = y_test_preds[i] == y_test[i]
    f2 = f1 and y_test_preds[i] != y_test_adv_preds[i]
    if targeted:
        f3 = f2 and y_test_adv_preds[i] == y_test_adv[i]
    else:
        f3 = f2
    if f1:
        f1_inds.append(i)
    else:
        f0_inds.append(i)
    if f2:
        f2_inds.append(i)
    if f3:
        f3_inds.append(i)

f0_inds = np.asarray(f0_inds)
f1_inds = np.asarray(f1_inds)
f2_inds = np.asarray(f2_inds)
f3_inds = np.asarray(f3_inds)

print("Number of test samples: {}. #net_succ: {}. #net_succ_attack_flip: {}. # net_succ_attack_succ: {}"
      .format(test_size, len(f1_inds), len(f2_inds), len(f3_inds)))

if args.norm == 'L_inf':
    norm = np.inf
elif args.norm == 'L1':
    norm = 1
elif args.norm == 'L2':
    norm = 2
else:
    raise AssertionError('norm {} is not acceptible'.format(args.norm))

explorer = BallExplorer(
    classifier=classifier,
    rand_gen=rand_gen,
    norm=norm,
    eps=args.eps,
    num_points=args.num_points,
    batch_size=batch_size,
    wlg=args.wlg,
    wpg=args.wpg
)

print('calculating original gradients in ball...')
# x_ball, losses, preds, losses_grads, preds_grads = explorer.generate(X_test)
x_ball, losses, preds, _, _ = explorer.generate(X_test)

print('calculating adv gradients in ball...')
# x_ball_adv, losses_adv, preds_adv, losses_grads_adv, preds_grads_adv = explorer.generate(X_test_adv)
x_ball_adv, losses_adv, preds_adv, _, _ = explorer.generate(X_test_adv)
print('done')

# first, for each image sort all the samples by norm distance from the main
x_dist     = np.empty((test_size, args.num_points), dtype=np.float32)
x_dist_adv = np.empty((test_size, args.num_points), dtype=np.float32)
for j in range(args.num_points):
    x_dist[:, j]     = np.linalg.norm((x_ball[:, j] - X_test).reshape((test_size, -1)), axis=1, ord=norm)
    x_dist_adv[:, j] = np.linalg.norm((x_ball_adv[:, j] - X_test_adv).reshape((test_size, -1)), axis=1, ord=norm)
ranks     = x_dist.argsort(axis=1)
ranks_adv = x_dist_adv.argsort(axis=1)

# sorting the points in the ball
for i in range(test_size):
    rks     = ranks[i]
    rks_adv = ranks_adv[i]

    x_ball[i]           = x_ball[i, rks]
    losses[i]           = losses[i, rks]
    preds[i]            = preds[i, rks]
    # losses_grads[i]     = losses_grads[i, rks]
    # preds_grads[i]      = preds_grads[i, rks]
    x_dist[i]           = x_dist[i, rks]

    x_ball_adv[i]       = x_ball_adv[i, rks_adv]
    losses_adv[i]       = losses_adv[i, rks_adv]
    preds_adv[i]        = preds_adv[i, rks_adv]
    # losses_grads_adv[i] = losses_grads_adv[i, rks_adv]
    # preds_grads_adv[i]  = preds_grads_adv[i, rks_adv]
    x_dist_adv[i]       = x_dist_adv[i, rks_adv]

# calculating softmax predictions:
probs     = scipy.special.softmax(preds, axis=2)
probs_adv = scipy.special.softmax(preds_adv, axis=2)

# converting everything from 3x32x32 to 32x32x3
X_test_img     = convert_tensor_to_image(X_test)
X_test_adv_img = convert_tensor_to_image(X_test_adv)
x_ball_img     = convert_tensor_to_image(x_ball.reshape((test_size * args.num_points, ) + X_test.shape[1:])) \
                .reshape((test_size, args.num_points) + X_test_img.shape[1:])
x_ball_adv_img = convert_tensor_to_image(x_ball_adv.reshape((test_size * args.num_points, ) + X_test.shape[1:])) \
                .reshape((test_size, args.num_points) + X_test_img.shape[1:])

# visualizing the images in the ball
n_imgs = 5   # number of images
n_dist = 10  # number of distortions
assert args.num_points % n_dist == 0
p_delta = int(args.num_points / n_dist)
inds = rand_gen.choice(f3_inds, n_imgs, replace=False)
fig = plt.figure(figsize=(n_dist, 2 * n_imgs))
for i in range(n_imgs):
    for p in range(n_dist):
        loc = n_dist * (2 * i) + p + 1
        fig.add_subplot(2 * n_imgs, n_dist, loc)
        plt.imshow(x_ball_img[inds[i], p * p_delta])
        plt.axis('off')
        loc = n_dist * (2 * i + 1) + p + 1
        fig.add_subplot(2 * n_imgs, n_dist, loc)
        plt.imshow(x_ball_adv_img[inds[i], p * p_delta])
        plt.axis('off')
plt.tight_layout()
plt.show()

i = inds[1]

# plotting loss
plt.figure()
plt.plot(100.0 * (losses[i] - losses[i, 0])/losses[i, 0])
plt.title('Loss for x in ball vs L2 distance from original x. i={}'.format(i))
plt.show()

plt.figure()
plt.plot(100.0 * (losses_adv[i] - losses_adv[i, 0])/losses_adv[i, 0])
plt.title('Loss for x in ball vs L2 distance from original x_adv. i={}'.format(i))
plt.show()

# plotting probabilities raw values
n_cols = 5
n_rows = int(np.ceil(len(classes) / n_cols))
fig = plt.figure(figsize=(10 * n_cols, 10 * n_rows))
for c in range(len(classes)):
    loc = c + 1
    ax1 = fig.add_subplot(n_rows, n_cols, loc)
    ax1.set_xlabel('noise rank')
    ax1.set_ylabel('normal probs[{}]'.format(c), color='blue')
    ax1.set_ylim(-0.05, 1.05)
    ax1.plot(probs[i, :, c], color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('adv probs[{}]'.format(c), color='red')
    ax2.set_ylim(-0.05, 1.05)
    ax2.plot(probs_adv[i, :, c], color='red')
    ax2.tick_params(axis='y', labelcolor='red')
plt.tight_layout()
fig.suptitle('raw probs values for i={}'.format(i))
plt.show()



# plotting preds raw values
n_cols = 5
n_rows = int(np.ceil(len(classes) / n_cols))
fig = plt.figure(figsize=(10 * n_cols, 10 * n_rows))
for c in range(len(classes)):
    loc = c + 1
    ax1 = fig.add_subplot(n_rows, n_cols, loc)
    ax1.set_xlabel('noise rank')
    ax1.set_ylabel('normal pred[{}]'.format(c), color='blue')
    ax1.plot(preds[i, :, c], color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('adv pred[{}]'.format(c), color='red')
    ax2.plot(preds_adv[i, :, c], color='red')
    ax2.tick_params(axis='y', labelcolor='red')
plt.tight_layout()
fig.suptitle('raw preds values for i={}'.format(i))
plt.show()

# plotting preds relative values
n_cols = 5
n_rows = int(np.ceil(len(classes) / n_cols))
fig = plt.figure(figsize=(10 * n_cols, 10 * n_rows))
for c in range(len(classes)):
    loc = c + 1
    ax1 = fig.add_subplot(n_rows, n_cols, loc)
    ax1.set_xlabel('noise rank')
    ax1.set_ylabel('normal relative pred[{}]'.format(c), color='blue')
    ax1.plot(100.0 * (preds[i, :, c] - preds[i, 0, c]) / np.abs(preds[i, 0, c]), color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('adv relative pred[{}]'.format(c), color='red')
    ax2.plot(100.0 * (preds_adv[i, :, c] - preds_adv[i, 0, c]) / np.abs(preds_adv[i, 0, c]), color='red')
    ax2.tick_params(axis='y', labelcolor='red')
plt.tight_layout()
fig.suptitle('relative preds values for i={}'.format(i))
plt.show()

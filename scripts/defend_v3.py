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
from active_learning_project.utils import convert_tensor_to_image, boolean_string, majority_vote

import matplotlib.pyplot as plt

from art.classifiers import PyTorchClassifier
from active_learning_project.classifiers.pytorch_ext_classifier import PyTorchExtClassifier


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='deepfool', type=str, help='attack directory')
parser.add_argument('--save_dir', default='ball_rev_L2_eps_8_n_1000', type=str, help='reverse dir')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--subset', default=-1, type=int, help='attack only subset of test set')

# for exploration
parser.add_argument('--norm', default='L2', type=str, help='norm or ball distance')
parser.add_argument('--eps', default=8.0, type=float, help='the ball radius for exploration')
parser.add_argument('--num_points', default=1000, type=int, help='the number of gradients to sample')
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

if not os.path.exists(os.path.join(SAVE_DIR, 'x_ball.npy')):
    print('calculating normal x in ball...')
    x_ball, losses, preds = explorer.generate(X_test)

    # print('calculating adv x in ball...')
    # x_ball_adv, losses_adv, preds_adv = explorer.generate(X_test_adv)
    # print('done calculating x ball')

    # first, for each image sort all the samples by norm distance from the main
    x_dist     = np.empty((test_size, args.num_points), dtype=np.float32)
    # x_dist_adv = np.empty((test_size, args.num_points), dtype=np.float32)
    for j in range(args.num_points):
        x_dist[:, j]     = np.linalg.norm((x_ball[:, j] - X_test).reshape((test_size, -1)), axis=1, ord=norm)
        # x_dist_adv[:, j] = np.linalg.norm((x_ball_adv[:, j] - X_test_adv).reshape((test_size, -1)), axis=1, ord=norm)
    ranks     = x_dist.argsort(axis=1)
    # ranks_adv = x_dist_adv.argsort(axis=1)

    # sorting the points in the ball
    for i in range(test_size):
        rks     = ranks[i]
        # rks_adv = ranks_adv[i]

        x_ball[i]           = x_ball[i, rks]
        losses[i]           = losses[i, rks]
        preds[i]            = preds[i, rks]
        x_dist[i]           = x_dist[i, rks]

        # x_ball_adv[i]       = x_ball_adv[i, rks_adv]
        # losses_adv[i]       = losses_adv[i, rks_adv]
        # preds_adv[i]        = preds_adv[i, rks_adv]
        # x_dist_adv[i]       = x_dist_adv[i, rks_adv]

    print('start saving to disk ({})...'.format(SAVE_DIR))
    np.save(os.path.join(SAVE_DIR, 'x_ball.npy'), x_ball)
    np.save(os.path.join(SAVE_DIR, 'losses.npy'), losses)
    np.save(os.path.join(SAVE_DIR, 'preds.npy'), preds)
    np.save(os.path.join(SAVE_DIR, 'x_dist.npy'), x_dist)
    # np.save(os.path.join(SAVE_DIR, 'x_ball_adv.npy'), x_ball_adv)
    # np.save(os.path.join(SAVE_DIR, 'losses_adv.npy'), losses_adv)
    # np.save(os.path.join(SAVE_DIR, 'preds_adv.npy'), preds_adv)
    # np.save(os.path.join(SAVE_DIR, 'x_dist_adv.npy'), x_dist_adv)
else:
    x_ball     = np.load(os.path.join(SAVE_DIR, 'x_ball.npy'))
    losses     = np.load(os.path.join(SAVE_DIR, 'losses.npy'))
    preds      = np.load(os.path.join(SAVE_DIR, 'preds.npy'))
    x_dist     = np.load(os.path.join(SAVE_DIR, 'x_dist.npy'))
    x_ball_adv = np.load(os.path.join(SAVE_DIR, 'x_ball_adv.npy'))
    losses_adv = np.load(os.path.join(SAVE_DIR, 'losses_adv.npy'))
    preds_adv  = np.load(os.path.join(SAVE_DIR, 'preds_adv.npy'))
    x_dist_adv = np.load(os.path.join(SAVE_DIR, 'x_dist_adv.npy'))

exit(0)
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

i = inds[0]

# get useful stats:
rel_losses     = np.zeros_like(losses)
rel_losses_adv = np.zeros_like(losses)
for i in range(test_size):
    rel_losses[i]     = (losses[i] - losses[i, 0])/losses[i, 0]
    rel_losses_adv[i] = (losses_adv[i] - losses_adv[i, 0])/losses_adv[i, 0]

switch_ranks     = []
switch_ranks_adv = []
y_ball_preds     = probs.argmax(axis=2)
y_ball_adv_preds = probs_adv.argmax(axis=2)
for i in range(test_size):
    rks = np.where(y_ball_preds[i] != y_test_preds[i])[0]
    switch_ranks.append(rks)
    rks = np.where(y_ball_adv_preds[i] != y_test_adv_preds[i])[0]
    switch_ranks_adv.append(rks)

# plotting loss
plt.figure()
plt.plot(losses[i])
plt.title('Raw loss for x in ball vs L2 distance from original normal x. i={}'.format(i))
plt.show()

plt.figure()
plt.plot(losses_adv[i], 'red')
plt.title('Raw loss for x in ball vs L2 distance from original x_adv x. i={}'.format(i))
plt.show()

plt.figure()
plt.plot(losses[i], 'blue')
plt.plot(losses_adv[i], 'red')
plt.title('Raw losses for x in ball vs L2 distance from original. i={}'.format(i))
plt.legend(['normal', 'adv'])
plt.show()

# plotting relative loss
plt.figure()
plt.plot(rel_losses[i], 'blue')
plt.title('Relative loss for x in ball vs L2 distance from original x. i={}'.format(i))
plt.show()

plt.figure()
plt.plot(rel_losses_adv[i], 'red')
plt.title('Relative loss for x in ball vs L2 distance from original x_adv. i={}'.format(i))
plt.show()

plt.figure()
plt.plot(rel_losses[i], 'blue')
plt.plot(rel_losses_adv[i], 'red')
plt.title('Relative loss for x in ball vs L2 distance from original. i={}'.format(i))
plt.legend(['normal', 'adv'])
plt.show()

# Finding a single value to threshold for norm/adv.
# guess #1: integral(loss)
intg_losses     = np.sum(losses, axis=1)
intg_losses_adv = np.sum(losses_adv, axis=1)
plt.figure()
plt.hist(intg_losses[f3_inds], alpha=0.5, label='normal', bins=10)
plt.hist(intg_losses_adv[f3_inds], alpha=0.5, label='adv', bins=10)
plt.legend(loc='upper right')
# plt.ylim(0, 0.005)
plt.show()

# guess #2: integral(relative loss)
intg_rel_losses     = np.sum(rel_losses, axis=1)
intg_rel_losses_adv = np.sum(rel_losses_adv, axis=1)

plt.figure()
plt.hist(intg_rel_losses[f3_inds], alpha=0.5, label='normal', bins=10)
plt.hist(intg_rel_losses_adv[f3_inds], alpha=0.5, label='adv', bins=10)
plt.legend(loc='upper right')
# plt.xlim(-300, 500000)
# plt.ylim(0, 500)
plt.show()

# guess #3: max(relative loss)
max_rel_losses     = np.max(rel_losses, axis=1)
max_rel_losses_adv = np.max(rel_losses_adv, axis=1)

plt.figure()
plt.hist(max_rel_losses[f3_inds], alpha=0.5, label='normal', bins=10)
plt.hist(max_rel_losses_adv[f3_inds], alpha=0.5, label='adv', bins=10)
plt.legend(loc='upper right')
# plt.xlim(-300, 500000)
plt.show()

# guess #3.1: max(loss)
max_losses     = np.max(losses, axis=1)
max_losses_adv = np.max(losses_adv, axis=1)

plt.figure()
plt.hist(max_losses[f3_inds], alpha=0.5, label='normal', bins=10)
plt.hist(max_losses_adv[f3_inds], alpha=0.5, label='adv', bins=10)
plt.legend(loc='upper right')
# plt.xlim(-300, 500000)
plt.show()

# guess #4: rank @rel_loss= thd * orig rel_loss
top_rank     = -10 * np.ones(test_size)
top_rank_adv = -10 * np.ones(test_size)
for i in range(test_size):
    max_val = np.max(rel_losses[i])
    if max_val == 0.0:
        continue
    thd_val = 0.2 * max_val
    top_rank[i] = np.argmax(rel_losses[i] > thd_val)

    max_val = np.max(rel_losses_adv[i])
    if max_val == 0.0:
        continue
    thd_val = 0.2 * max_val
    top_rank_adv[i] = np.argmax(rel_losses_adv[i] > thd_val)

plt.figure()
plt.hist(top_rank[f3_inds], alpha=0.5, label='normal', bins=10)
plt.hist(top_rank_adv[f3_inds], alpha=0.5, label='adv', bins=10)
plt.legend(loc='upper right')
# plt.xlim(-300, 500000)
plt.show()

# guess 5: detection. integral(loss) from rank=0 until rank=@first pred switch
first_sw_rank     = -10 * np.ones(test_size, dtype=np.int32)
first_sw_rank_adv = -10 * np.ones(test_size, dtype=np.int32)
for i in range(test_size):
    if switch_ranks[i].size != 0:
        first_sw_rank[i] = switch_ranks[i][0]
    if switch_ranks_adv[i].size != 0:
        first_sw_rank_adv[i] = switch_ranks_adv[i][0]

plt.figure()
plt.hist(first_sw_rank[f3_inds], alpha=0.5, label='normal', bins=10)
plt.hist(first_sw_rank_adv[f3_inds], alpha=0.5, label='adv', bins=10)
plt.legend(loc='upper right')
# plt.xlim(-300, 500000)
plt.show()

# guess 6: detection. rank=@first pred switch
num_switches     = np.empty(test_size)
num_switches_adv = np.empty(test_size)
for i in range(test_size):
    num_switches[i]     = np.where(y_ball_preds[i] != y_test_preds[i])[0].size
    num_switches_adv[i] = np.where(y_ball_adv_preds[i] != y_test_adv_preds[i])[0].size

plt.figure()
plt.hist(num_switches[f3_inds], alpha=0.5, label='normal', bins=100)
plt.hist(num_switches_adv[f3_inds], alpha=0.5, label='adv', bins=100)
plt.legend(loc='upper right')
# plt.xlim(-300, 500000)
plt.show()

# guess 7: integral(sw) only for switched ranks
switch_ranks     = []
switch_ranks_adv = []
y_ball_preds     = probs.argmax(axis=2)
y_ball_adv_preds = probs_adv.argmax(axis=2)
for i in range(test_size):
    rks = np.where(y_ball_preds[i] != y_test_preds[i])[0]
    switch_ranks.append(rks)
    rks = np.where(y_ball_adv_preds[i] != y_test_adv_preds[i])[0]
    switch_ranks_adv.append(rks)

intg_sw     = np.zeros(test_size)
intg_sw_adv = np.zeros(test_size)
for i in range(test_size):
    for sw in switch_ranks[i]:
        intg_sw[i] += sw
    for sw in switch_ranks_adv[i]:
        intg_sw_adv[i] += sw

plt.figure()
plt.hist(intg_sw[f3_inds], alpha=0.5, label='normal', bins=100)
plt.hist(intg_sw_adv[f3_inds], alpha=0.5, label='adv', bins=100)
plt.legend(loc='upper right')
# plt.xlim(-300, 500000)
plt.show()

# guess 8: integral(loss) only for switched ranks
intg_loss_sw     = np.zeros(test_size)
intg_loss_sw_adv = np.zeros(test_size)
for i in range(test_size):
    for sw in switch_ranks[i]:
        intg_loss_sw[i] += losses[i, sw]
    for sw in switch_ranks_adv[i]:
        intg_loss_sw_adv[i] += losses_adv[i, sw]

plt.figure()
plt.hist(intg_loss_sw[f3_inds], alpha=0.5, label='normal', bins=100)
plt.hist(intg_loss_sw_adv[f3_inds], alpha=0.5, label='adv', bins=100)
plt.legend(loc='upper right')
# plt.xlim(-300, 500000)
plt.show()

# guess 9: integral(rel_loss) only for switched ranks
intg_rel_loss_sw     = np.zeros(test_size)
intg_rel_loss_sw_adv = np.zeros(test_size)
for i in range(test_size):
    for sw in switch_ranks[i]:
        intg_rel_loss_sw[i] += rel_losses[i, sw]
    for sw in switch_ranks_adv[i]:
        intg_rel_loss_sw_adv[i] += rel_losses_adv[i, sw]

plt.figure()
plt.hist(intg_rel_loss_sw[f3_inds], alpha=0.5, label='normal', bins=100, range=[0.001, 1e7])
plt.hist(intg_rel_loss_sw_adv[f3_inds], alpha=0.5, label='adv', bins=100, range=[0.001, 1e7])
plt.legend(loc='upper right')
# plt.xlim(-300, 500000)
plt.show()

# guess 10: integral(loss * prob) only for switched ranks
intg_loss_prob_sw     = np.zeros(test_size)
intg_loss_prob_sw_adv = np.zeros(test_size)
for i in range(test_size):
    orig_pred = y_test_preds[i]
    for sw in switch_ranks[i]:
        intg_loss_prob_sw[i] += (losses[i, sw] * probs[i, sw, orig_pred])
    orig_pred = y_test_adv_preds[i]
    for sw in switch_ranks_adv[i]:
        intg_loss_prob_sw_adv[i] += (losses_adv[i, sw] * probs_adv[i, sw, orig_pred])

plt.figure()
plt.hist(intg_loss_prob_sw[f3_inds], alpha=0.5, label='normal', bins=100, range=[0.001, 60])
plt.hist(intg_loss_prob_sw_adv[f3_inds], alpha=0.5, label='adv', bins=100, range=[0.001, 60])
plt.legend(loc='upper right')
# plt.xlim(-300, 500000)
plt.show()

# guess 11: integral(rel_loss * prob) only for switched ranks
intg_rel_loss_prob_sw     = np.zeros(test_size)
intg_rel_loss_prob_sw_adv = np.zeros(test_size)
for i in range(test_size):
    orig_pred = y_test_preds[i]
    for sw in switch_ranks[i]:
        intg_rel_loss_prob_sw[i] += (rel_losses[i, sw] * probs[i, sw, orig_pred])
    orig_pred = y_test_adv_preds[i]
    for sw in switch_ranks_adv[i]:
        intg_rel_loss_prob_sw_adv[i] += (rel_losses_adv[i, sw] * probs_adv[i, sw, orig_pred])

plt.figure()
plt.hist(intg_rel_loss_prob_sw[f3_inds], alpha=0.5, label='normal', bins=100, range=[0.00001, 2000], density=True)
plt.hist(intg_rel_loss_prob_sw_adv[f3_inds], alpha=0.5, label='adv', bins=100, range=[0.00001, 2000], density=True)
plt.legend(loc='upper right')
# plt.xlim(-300, 500000)
plt.show()

# guess 12: cum_sum loss
cum_sum_losses     = np.cumsum(losses, axis=1)
cum_sum_losses_adv = np.cumsum(losses_adv, axis=1)

plt.figure()
plt.hist(cum_sum_losses[f3_inds], alpha=0.5, label='normal', bins=100)
plt.hist(cum_sum_losses_adv[f3_inds], alpha=0.5, label='adv', bins=100)
plt.legend(loc='upper right')
# plt.xlim(-300, 500000)
plt.show()












# guess #6: robustness. find label by integrating probs
intg_probs     = np.sum(probs, axis=1)
intg_probs_adv = np.sum(probs_adv, axis=1)
defense_preds     = intg_probs.argmax(axis=1)
defense_preds_adv = intg_probs_adv.argmax(axis=1)

# guess #6: robustness. find label by integrating preds
intg_preds     = np.sum(preds, axis=1)
intg_preds_adv = np.sum(preds_adv, axis=1)
defense_preds     = intg_preds.argmax(axis=1)
defense_preds_adv = intg_preds_adv.argmax(axis=1)

acc_all = np.mean(defense_preds == y_test)
acc_f1 = np.mean(defense_preds[f1_inds] == y_test[f1_inds])
acc_f2 = np.mean(defense_preds[f2_inds] == y_test[f2_inds])
acc_f3 = np.mean(defense_preds[f3_inds] == y_test[f3_inds])

acc_all_adv = np.mean(defense_preds_adv == y_test)
acc_f1_adv = np.mean(defense_preds_adv[f1_inds] == y_test[f1_inds])
acc_f2_adv = np.mean(defense_preds_adv[f2_inds] == y_test[f2_inds])
acc_f3_adv = np.mean(defense_preds_adv[f3_inds] == y_test[f3_inds])

print('Accuracy: all samples: {:.2f}/{:.2f}%, f1 samples: {:.2f}/{:.2f}%, f2 samples: {:.2f}/{:.2f}%, f3 samples: {:.2f}/{:.2f}%'
      .format(acc_all * 100, acc_all_adv * 100, acc_f1 * 100, acc_f1_adv * 100, acc_f2 * 100, acc_f2_adv * 100, acc_f3 * 100, acc_f3_adv * 100))









# # plotting probabilities raw values
# n_cols = 5
# n_rows = int(np.ceil(len(classes) / n_cols))
# fig = plt.figure(figsize=(10 * n_cols, 10 * n_rows))
# for c in range(len(classes)):
#     loc = c + 1
#     ax1 = fig.add_subplot(n_rows, n_cols, loc)
#     ax1.set_xlabel('noise rank')
#     ax1.set_ylabel('normal probs[{}]'.format(c), color='blue')
#     ax1.set_ylim(-0.05, 1.05)
#     ax1.plot(probs[i, :, c], color='blue')
#     ax1.tick_params(axis='y', labelcolor='blue')
#
#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#     ax2.set_ylabel('adv probs[{}]'.format(c), color='red')
#     ax2.set_ylim(-0.05, 1.05)
#     ax2.plot(probs_adv[i, :, c], color='red')
#     ax2.tick_params(axis='y', labelcolor='red')
# plt.tight_layout()
# fig.suptitle('raw probs values for i={}'.format(i))
# plt.show()

# plotting preds raw values
# n_cols = 5
# n_rows = int(np.ceil(len(classes) / n_cols))
# fig = plt.figure(figsize=(10 * n_cols, 10 * n_rows))
# for c in range(len(classes)):
#     loc = c + 1
#     ax1 = fig.add_subplot(n_rows, n_cols, loc)
#     ax1.set_xlabel('noise rank')
#     ax1.set_ylabel('normal pred[{}]'.format(c), color='blue')
#     ax1.plot(preds[i, :, c], color='blue')
#     ax1.tick_params(axis='y', labelcolor='blue')
#
#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#     ax2.set_ylabel('adv pred[{}]'.format(c), color='red')
#     ax2.plot(preds_adv[i, :, c], color='red')
#     ax2.tick_params(axis='y', labelcolor='red')
# plt.tight_layout()
# fig.suptitle('raw preds values for i={}'.format(i))
# plt.show()

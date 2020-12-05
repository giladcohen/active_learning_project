import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms

import PIL
from tqdm import tqdm
import numpy as np
import json
import os
import argparse
import sys
import scipy
from datetime import datetime
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from active_learning_project.attacks.tta_ball_explorer import TTABallExplorer
from active_learning_project.utils import convert_tensor_to_image, boolean_string, majority_vote, add_feature

import matplotlib.pyplot as plt

from art.classifiers import PyTorchClassifier
from active_learning_project.classifiers.pytorch_ext_classifier import PyTorchExtClassifier

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='deepfool', type=str, help='attack directory')
parser.add_argument('--save_dir', default='tta_ball_rev_L2_eps_8_n_1000', type=str, help='reverse dir')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--subset', default=-1, type=int, help='attack only subset of test set')

# for exploration
parser.add_argument('--norm', default='L2', type=str, help='norm or ball distance')
parser.add_argument('--eps', default=8.0, type=float, help='the ball radius for exploration')
parser.add_argument('--num_points', default=1000, type=int, help='the number of gradients to sample')

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
all_inds = np.arange(test_size)

print("Number of test samples: {}. #net_succ: {}. #net_succ_attack_flip: {}. # net_succ_attack_succ: {}"
      .format(test_size, len(f1_inds), len(f2_inds), len(f3_inds)))

# dividing the official test set to a val set and to a test set
val_inds = rand_gen.choice(all_inds, int(0.5*test_size), replace=False)
val_inds.sort()
f0_inds_val = np.asarray([ind for ind in f0_inds if ind in val_inds])
f1_inds_val = np.asarray([ind for ind in f1_inds if ind in val_inds])
f2_inds_val = np.asarray([ind for ind in f2_inds if ind in val_inds])
f3_inds_val = np.asarray([ind for ind in f3_inds if ind in val_inds])

test_inds = np.asarray([ind for ind in all_inds if ind not in val_inds])
f0_inds_test = np.asarray([ind for ind in f0_inds if ind in test_inds])
f1_inds_test = np.asarray([ind for ind in f1_inds if ind in test_inds])
f2_inds_test = np.asarray([ind for ind in f2_inds if ind in test_inds])
f3_inds_test = np.asarray([ind for ind in f3_inds if ind in test_inds])

if args.norm == 'L_inf':
    norm = np.inf
elif args.norm == 'L1':
    norm = 1
elif args.norm == 'L2':
    norm = 2
else:
    raise AssertionError('norm {} is not acceptible'.format(args.norm))

# DEBUG:
# X_test_img     = convert_tensor_to_image(X_test)
# X_tta_test     = get_normalized_tensor(tta_testloader, batch_size)
# X_tta_test_img = convert_tensor_to_image(X_tta_test)
#
# plt.imshow(X_test_img[14])
# plt.show()
# plt.imshow(X_tta_test_img[14])
# plt.show()

explorer = TTABallExplorer(
    classifier=classifier,
    dataset=train_args['dataset'],
    rand_gen=rand_gen,
    norm=norm,
    eps=args.eps,
    num_points=args.num_points,
    batch_size=batch_size,
)

if not os.path.exists(os.path.join(SAVE_DIR, 'x_ball_adv_subset_500.npy')):
    print('calculating normal x in ball...')
    x_ball, losses, preds, noise_powers = explorer.generate(X_test)
    print('done calculating x ball')

    ranks = noise_powers.argsort(axis=1)

    # sorting the points in the ball
    for i in range(test_size):
        rks = ranks[i]
        x_ball[i]       = x_ball[i, rks]
        losses[i]       = losses[i, rks]
        preds[i]        = preds[i, rks]
        noise_powers[i] = noise_powers[i, rks]

    print('start saving to disk ({})...'.format(SAVE_DIR))
    np.save(os.path.join(SAVE_DIR, 'x_ball_subset_500.npy'), x_ball[0:500])
    np.save(os.path.join(SAVE_DIR, 'losses.npy'), losses)
    np.save(os.path.join(SAVE_DIR, 'preds.npy'), preds)
    np.save(os.path.join(SAVE_DIR, 'noise_powers.npy'), noise_powers)

    x_ball = x_ball[0:500]  # expensive in memory

    print('calculating adv x in ball...')
    x_ball_adv, losses_adv, preds_adv, noise_powers_adv = explorer.generate(X_test_adv)
    print('done calculating x adv ball')

    ranks_adv = noise_powers_adv.argsort(axis=1)

    for i in range(test_size):
        rks_adv = ranks_adv[i]
        x_ball_adv[i]       = x_ball_adv[i, rks]
        losses_adv[i]       = losses_adv[i, rks]
        preds_adv[i]        = preds_adv[i, rks]
        noise_powers_adv[i] = noise_powers_adv[i, rks]

    print('start saving to disk ({})...'.format(SAVE_DIR))
    np.save(os.path.join(SAVE_DIR, 'x_ball_adv_subset_500.npy'), x_ball_adv[0:500])
    np.save(os.path.join(SAVE_DIR, 'losses_adv.npy'), losses_adv)
    np.save(os.path.join(SAVE_DIR, 'preds_adv.npy'), preds_adv)
    np.save(os.path.join(SAVE_DIR, 'noise_powers_adv.npy'), noise_powers_adv)

    x_ball_adv = x_ball_adv[0:500]  # expensive in memory

    print('done')
    exit(0)
else:
    x_ball     = np.load(os.path.join(SAVE_DIR, 'x_ball_subset_500.npy'))
    losses     = np.load(os.path.join(SAVE_DIR, 'losses.npy'))
    preds      = np.load(os.path.join(SAVE_DIR, 'preds.npy'))
    x_dist     = np.load(os.path.join(SAVE_DIR, 'noise_powers.npy'))
    x_ball_adv = np.load(os.path.join(SAVE_DIR, 'x_ball_adv_subset_500.npy'))
    losses_adv = np.load(os.path.join(SAVE_DIR, 'losses_adv.npy'))
    preds_adv  = np.load(os.path.join(SAVE_DIR, 'preds_adv.npy'))
    x_dist_adv = np.load(os.path.join(SAVE_DIR, 'noise_powers_adv.npy'))

# calculating softmax predictions:
probs     = scipy.special.softmax(preds, axis=2)
probs_adv = scipy.special.softmax(preds_adv, axis=2)

# converting everything from 3x32x32 to 32x32x3
X_test_img     = convert_tensor_to_image(X_test)
X_test_adv_img = convert_tensor_to_image(X_test_adv)
# x_ball_img     = convert_tensor_to_image(x_ball.reshape((test_size * args.num_points, ) + X_test.shape[1:])) \
#                 .reshape((test_size, args.num_points) + X_test_img.shape[1:])
x_ball_img     = convert_tensor_to_image(x_ball.reshape((500 * args.num_points, ) + X_test.shape[1:])) \
                .reshape((500, args.num_points) + X_test_img.shape[1:])
# x_ball_adv_img = convert_tensor_to_image(x_ball_adv.reshape((test_size * args.num_points, ) + X_test.shape[1:])) \
#                 .reshape((test_size, args.num_points) + X_test_img.shape[1:])
x_ball_adv_img = convert_tensor_to_image(x_ball_adv.reshape((500 * args.num_points, ) + X_test.shape[1:])) \
                .reshape((500, args.num_points) + X_test_img.shape[1:])

# visualizing the images in the ball
n_imgs = 5   # number of images
n_dist = 10  # number of distortions
assert args.num_points % n_dist == 0
p_delta = int(args.num_points / n_dist)
inds = rand_gen.choice([si for si in f3_inds if si < 500], n_imgs, replace=False)
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

confidences = np.max(probs, axis=2)
confidences_adv = np.max(probs_adv, axis=2)

# get images index without any switched pred:
no_sw_pred_inds = []
no_sw_pred_inds_adv = []
for i in range(test_size):
    if switch_ranks[i].size == 0:
        no_sw_pred_inds.append(i)
    if switch_ranks_adv[i].size == 0:
        no_sw_pred_inds_adv.append(i)

# setting SVM classifier and init features
normal_features = []
adv_features = []
features_index = []

# plotting loss
i = inds[0]
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
intg_losses     = np.cumsum(losses, axis=1)
intg_losses_adv = np.cumsum(losses_adv, axis=1)

plt.figure()
plt.hist(intg_losses[f3_inds, 399], alpha=0.5, label='normal', bins=100)
plt.hist(intg_losses_adv[f3_inds, 399], alpha=0.5, label='adv', bins=100)
plt.legend(loc='upper right')
plt.title('intg_loss_up_to_rank_400')
# plt.ylim(0, 0.005)
plt.show()

features_index.append('intg_loss_up_to_rank_400')
normal_features.append(intg_losses[:, 399])
adv_features.append(intg_losses_adv[:, 399])

# guess #2: integral(relative loss)
intg_rel_losses     = np.sum(rel_losses, axis=1)
intg_rel_losses_adv = np.sum(rel_losses_adv, axis=1)

plt.figure()
plt.hist(intg_rel_losses[f3_inds], alpha=0.5, label='normal', bins=100)#, range=[-950, 1e5])
plt.hist(intg_rel_losses_adv[f3_inds], alpha=0.5, label='adv', bins=100)#, range=[-950, 1e5])
plt.legend(loc='upper right')
plt.title('intg_relative_loss')
# plt.xlim(-300, 500000)
# plt.ylim(0, 500)
plt.show()

features_index.append('intg_relative_loss')
normal_features.append(intg_rel_losses)
adv_features.append(intg_rel_losses_adv)

# guess #3: max(relative loss)
max_rel_losses     = np.max(rel_losses, axis=1)
max_rel_losses_adv = np.max(rel_losses_adv, axis=1)

plt.figure()
plt.hist(max_rel_losses[f3_inds], alpha=0.5, label='normal', bins=100)#, range=[0, 100])
plt.hist(max_rel_losses_adv[f3_inds], alpha=0.5, label='adv', bins=100)#, range=[0, 100])
plt.legend(loc='upper right')
plt.title('max_relative_loss')
# plt.xlim(-300, 500000)
plt.show()

features_index.append('max_relative_loss')
normal_features.append(max_rel_losses)
adv_features.append(max_rel_losses_adv)

# guess #3.1: max(loss)
# max_losses     = np.max(losses, axis=1)
# max_losses_adv = np.max(losses_adv, axis=1)
#
# plt.figure()
# plt.hist(max_losses[f3_inds], alpha=0.5, label='normal', bins=100)
# plt.hist(max_losses_adv[f3_inds], alpha=0.5, label='adv', bins=100)
# plt.legend(loc='upper right')
# # plt.xlim(-300, 500000)
# plt.show()

# guess #4: rank @rel_loss= thd * max_rel_loss
thd = 0.001
top_rank     = -1 * np.ones(test_size)
top_rank_adv = -1 * np.ones(test_size)
for i in range(test_size):
    max_val = np.max(rel_losses[i])
    if max_val > 0.0:
        thd_val = thd * max_val
        top_rank[i] = np.argmax(rel_losses[i] > thd_val)
    else:
        print('normal image i={} does not have rel_loss > 0'.format(i))

    max_val = np.max(rel_losses_adv[i])
    if max_val > 0.0:
        thd_val = thd * max_val
        top_rank_adv[i] = np.argmax(rel_losses_adv[i] > thd_val)
    else:
        print('adv image i={} does not have rel_loss > 0'.format(i))

plt.figure()
plt.hist(top_rank[f3_inds], alpha=0.5, label='normal', bins=100)#, range=[0, 100])
plt.hist(top_rank_adv[f3_inds], alpha=0.5, label='adv', bins=100)#, range=[0, 100])
plt.legend(loc='upper right')
plt.title('rank @ {}*max_rel_loss'.format(thd))
# plt.xlim(-300, 500000)
plt.show()

features_index.append('rank @ {}*max_rel_loss'.format(thd))
normal_features.append(top_rank)
adv_features.append(top_rank_adv)

# guess 5: detection. rank=@first pred switch
first_sw_rank     = -1 * np.ones(test_size, dtype=np.int32)
first_sw_rank_adv = -1 * np.ones(test_size, dtype=np.int32)
for i in range(test_size):
    if switch_ranks[i].size != 0:
        first_sw_rank[i] = switch_ranks[i][0]
    if switch_ranks_adv[i].size != 0:
        first_sw_rank_adv[i] = switch_ranks_adv[i][0]

first_sw_rank_features = np.where(first_sw_rank == -1, 3 * args.num_points, first_sw_rank)
first_sw_rank_adv_features = np.where(first_sw_rank_adv == -1, 3 * args.num_points, first_sw_rank_adv)

plt.figure()
plt.hist(first_sw_rank_features[f3_inds], alpha=0.5, label='normal', bins=100)
plt.hist(first_sw_rank_adv_features[f3_inds], alpha=0.5, label='adv', bins=100)
plt.legend(loc='upper right')
plt.title('rank @ first pred switch')
# plt.xlim(-300, 500000)
plt.show()

features_index.append('rank @ first pred switch')
normal_features.append(first_sw_rank_features)
adv_features.append(first_sw_rank_adv_features)

# guess 6: number of prediction switches until a specific rank. cumulative.
num_switches_cum     = np.zeros((test_size, args.num_points), dtype=np.int32)
num_switches_cum_adv = np.zeros((test_size, args.num_points), dtype=np.int32)
for i in range(test_size):
    for sw_rnk in switch_ranks[i]:
        num_switches_cum[i, sw_rnk:] += 1
    for sw_rnk in switch_ranks_adv[i]:
        num_switches_cum_adv[i, sw_rnk:] += 1

# plot one example:
i = inds[0]
plt.figure()
plt.plot(num_switches_cum[i], 'blue')
plt.plot(num_switches_cum_adv[i], 'red')
plt.title('num_switches_cum for i={}'.format(i))
plt.legend(['normal', 'adv'])
plt.show()

plt.figure()
plt.hist(num_switches_cum[f3_inds, 399], alpha=0.5, label='normal', bins=100)#, range=[0, 0.07])
plt.hist(num_switches_cum_adv[f3_inds, 399], alpha=0.5, label='adv', bins=100)#, range=[0, 0.07])
plt.legend(loc='upper right')
plt.title('number of pred switches from orig, until rank 400')
# plt.xlim(-300, 500000)
plt.show()

features_index.append('number of pred switches from orig, until rank 400')
normal_features.append(num_switches_cum[:, 399])
adv_features.append(num_switches_cum_adv[:, 399])

# guess 6.1: integral(ranks) only for switched ranks
# intg_sw     = np.zeros(test_size, dtype=np.int32)
# intg_sw_adv = np.zeros(test_size, dtype=np.int32)
# for i in range(test_size):
#     for sw in switch_ranks[i]:
#         if sw <= 400:  # use fewer samples for best seperation
#             intg_sw[i] += sw
#     for sw in switch_ranks_adv[i]:
#         if sw <= 400:
#             intg_sw_adv[i] += sw
#
# plt.figure()
# plt.hist(intg_sw[f3_inds], alpha=0.5, label='normal', bins=100, range=[0, 2500])
# plt.hist(intg_sw_adv[f3_inds], alpha=0.5, label='adv', bins=100, range=[0, 2500])
# plt.legend(loc='upper right')
# # plt.xlim(-300, 500000)
# plt.show()

# # guess 6.2: integral(loss) only for switched ranks
# intg_loss_sw     = np.zeros(test_size)
# intg_loss_sw_adv = np.zeros(test_size)
# for i in range(test_size):
#     for sw in switch_ranks[i]:
#         if sw <= 400:
#             intg_loss_sw[i] += losses[i, sw]
#     for sw in switch_ranks_adv[i]:
#         if sw <= 400:
#             intg_loss_sw_adv[i] += losses_adv[i, sw]
#
# plt.figure()
# plt.hist(intg_loss_sw[f3_inds], alpha=0.5, label='normal', bins=100, range=[0, 750])
# plt.hist(intg_loss_sw_adv[f3_inds], alpha=0.5, label='adv', bins=100, range=[0, 750])
# plt.legend(loc='upper right')
# # plt.xlim(-300, 500000)
# plt.show()

# # guess 6.3: integral(rel_loss) only for switched ranks
# intg_rel_loss_sw     = np.zeros(test_size)
# intg_rel_loss_sw_adv = np.zeros(test_size)
# for i in range(test_size):
#     for sw in switch_ranks[i]:
#         # if sw < 400:
#             intg_rel_loss_sw[i] += rel_losses[i, sw]
#     for sw in switch_ranks_adv[i]:
#         # if sw < 400:
#             intg_rel_loss_sw_adv[i] += rel_losses_adv[i, sw]
#
# plt.figure()
# plt.hist(intg_rel_loss_sw[f3_inds], alpha=0.5, label='normal', bins=100, range=[0, 50000])
# plt.hist(intg_rel_loss_sw_adv[f3_inds], alpha=0.5, label='adv', bins=100, range=[0, 50000])
# plt.legend(loc='upper right')
# # plt.xlim(-300, 500000)
# plt.show()

# # guess 6.4: integral(loss) for only correct (as first) prediction.
# intg_loss_orig_pred     = np.zeros(test_size)
# intg_loss_orig_pred_adv = np.zeros(test_size)
# for i in range(test_size):
#     for j in range(args.num_points):
#         if j not in switch_ranks[i]:
#             # if j <= 400:
#             intg_loss_orig_pred[i] += losses[i, j]
#         if j not in switch_ranks_adv[i]:
#             # if j <= 400:
#             intg_loss_orig_pred_adv[i] += losses_adv[i, j]
#
# plt.figure()
# plt.hist(intg_loss_orig_pred[f3_inds], alpha=0.5, label='normal', bins=100) #, range=[0, 750])
# plt.hist(intg_loss_orig_pred_adv[f3_inds], alpha=0.5, label='adv', bins=100) #, range=[0, 750])
# plt.legend(loc='upper right')
# # plt.xlim(-300, 500000)
# plt.show()

# guess 7: integral(loss) for only correct (as first) prediction.
intg_rel_loss_orig_pred     = np.zeros(test_size)
intg_rel_loss_orig_pred_adv = np.zeros(test_size)
for i in range(test_size):
    for j in range(args.num_points):
        if j not in switch_ranks[i]:
            # if j <= 400:
            intg_rel_loss_orig_pred[i] += rel_losses[i, j]
        if j not in switch_ranks_adv[i]:
            # if j <= 400:
            intg_rel_loss_orig_pred_adv[i] += rel_losses_adv[i, j]

plt.figure()
plt.hist(intg_rel_loss_orig_pred[f3_inds], alpha=0.5, label='normal', bins=100)#, range=[-1000, 1e4])
plt.hist(intg_rel_loss_orig_pred_adv[f3_inds], alpha=0.5, label='adv', bins=100)#, range=[-1000, 1e4])
plt.legend(loc='upper right')
plt.title('intg_relative_loss only for correct (first) prediction')
# plt.ylim([0, 500])
# plt.xlim(-300, 500000)
plt.show()

features_index.append('intg_relative_loss only for correct (first) prediction')
normal_features.append(intg_rel_loss_orig_pred)
adv_features.append(intg_rel_loss_orig_pred_adv)

# guess 8: confidence === max(prob). overall
# plot confidence for one example:
i = inds[0]
plt.figure()
plt.plot(confidences[i], 'blue')
plt.plot(confidences_adv[i], 'red')
plt.title('confidences for i={}'.format(i))
plt.legend(['normal', 'adv'])
plt.show()

confidences_cumsum = np.cumsum(confidences, axis=1)
confidences_cumsum_adv = np.cumsum(confidences_adv, axis=1)
# plot confidence_cumsum for one example:
i = inds[0]
plt.figure()
plt.plot(confidences_cumsum[i], 'blue')
plt.plot(confidences_cumsum_adv[i], 'red')
plt.title('confidences for i={}'.format(i))
plt.legend(['normal', 'adv'])
plt.show()

plt.figure()
plt.hist(confidences_cumsum[f3_inds, 199], alpha=0.5, label='normal', bins=100)#, range=[199, 200])
plt.hist(confidences_cumsum_adv[f3_inds, 199], alpha=0.5, label='adv', bins=100)#, range=[199, 200])
plt.legend(loc='upper right')
plt.title('intg(confidence) for top label. overall. until rank 200')
# plt.xlim(-300, 500000)
plt.show()

features_index.append('intg(confidence) for top label. overall. until rank 200')
normal_features.append(confidences_cumsum[:, 199])
adv_features.append(confidences_cumsum_adv[:, 199])

# guess 9: intg of confidence === max(prob) for primary only. specific.
# plot confidence for one example:
confidences_primary     = np.empty((test_size, args.num_points))
confidences_primary_adv = np.empty((test_size, args.num_points))
for i in range(test_size):
    confidences_primary[i] = probs[i, :, y_test_preds[i]]
    confidences_primary_adv[i] = probs_adv[i, :, y_test_adv_preds[i]]

confidences_primary_cumsum = np.cumsum(confidences_primary, axis=1)
confidences_primary_cumsum_adv = np.cumsum(confidences_primary_adv, axis=1)

plt.figure()
plt.hist(confidences_primary_cumsum[f3_inds, 399], alpha=0.5, label='normal', bins=100)#, range=[399, 400])
plt.hist(confidences_primary_cumsum_adv[f3_inds, 399], alpha=0.5, label='adv', bins=100)#, range=[399, 400])
plt.legend(loc='upper right')
plt.title('intg(confidence) for primary label. specific. until rank 400')
# plt.xlim(-300, 500000)
plt.show()

features_index.append('intg(confidence) for primary label. specific. until rank 400')
normal_features.append(confidences_primary_cumsum[:, 399])
adv_features.append(confidences_primary_cumsum_adv[:, 399])

# guess 10: confidence === max(prob) for secondary. overall
confidences_secondary_overall     = np.empty((test_size, args.num_points), dtype=np.float32)
confidences_secondary_overall_adv = np.empty((test_size, args.num_points), dtype=np.float32)

for i in range(test_size):
    for j in range(args.num_points):
        confidences_secondary_overall[i, j]     = np.sort(probs[i, j])[-2]
        confidences_secondary_overall_adv[i, j] = np.sort(probs_adv[i, j])[-2]

confidences_secondary_overall_cumsum     = np.cumsum(confidences_secondary_overall, axis=1)
confidences_secondary_overall_cumsum_adv = np.cumsum(confidences_secondary_overall_adv, axis=1)

i = inds[0]
plt.figure()
plt.plot(confidences_secondary_overall_cumsum[i], 'blue')
plt.plot(confidences_secondary_overall_cumsum_adv[i], 'red')
plt.title('confidences_secondary_cumsum for i={}'.format(i))
plt.legend(['normal', 'adv'])
plt.show()

plt.figure()
plt.hist(confidences_secondary_overall_cumsum[f3_inds, 399], alpha=0.5, label='normal', bins=100)#, range=[270, 300])
plt.hist(confidences_secondary_overall_cumsum_adv[f3_inds, 399], alpha=0.5, label='adv', bins=100)#, range=[270, 300])
plt.legend(loc='upper right')
plt.title('intg(confidence) for secondary label. overall. until rank 400')
# plt.ylim([0, 500])
# plt.xlim(-300, 500000)
plt.show()

features_index.append('intg(confidence) for secondary label. overall. until rank 400')
normal_features.append(confidences_secondary_overall_cumsum[:, 399])
adv_features.append(confidences_secondary_overall_cumsum_adv[:, 399])

# guess 11: intg confidence === max(prob) for secondary only. specific.
# First, find the secondary label
secondary_preds     = -1 * np.ones(test_size, dtype=np.int32)
secondary_preds_adv = -1 * np.ones(test_size, dtype=np.int32)

for i in range(test_size):
    if first_sw_rank[i] != -1:
        secondary_preds[i] = y_ball_preds[i, first_sw_rank[i]]
    else:
        print('for normal example i={}, there are no prediction switch'.format(i))
    if first_sw_rank_adv[i] != -1:
        secondary_preds_adv[i] = y_ball_adv_preds[i, first_sw_rank_adv[i]]
    else:
        print('for adv example i={}, there are no prediction switch'.format(i))

confidences_secondary     = -1 * np.ones((test_size, args.num_points))
confidences_secondary_adv = -1 * np.ones((test_size, args.num_points))
for i in range(test_size):
    if first_sw_rank[i] != -1:
        confidences_secondary[i] = probs[i, :, secondary_preds[i]]
    if first_sw_rank_adv[i] != -1:
        confidences_secondary_adv[i] = probs_adv[i, :, secondary_preds_adv[i]]

confidences_secondary_cumsum     = np.cumsum(confidences_secondary, axis=1)
confidences_secondary_cumsum_adv = np.cumsum(confidences_secondary_adv, axis=1)
for i in range(test_size):
    if first_sw_rank[i] == -1:
        confidences_secondary_cumsum[i] = 0
    if first_sw_rank_adv[i] == -1:
        confidences_secondary_cumsum_adv[i] = 0

plt.figure()
plt.hist(confidences_secondary_cumsum[f3_inds, 399], alpha=0.5, label='normal', bins=100)#, range=[-1, 40])
plt.hist(confidences_secondary_cumsum_adv[f3_inds, 399], alpha=0.5, label='adv', bins=100)#, range=[-1, 40])
plt.legend(loc='upper right')
plt.title('intg(confidence) for secondary label. specific. until rank 400')
# plt.xlim(-300, 500000)
plt.show()

features_index.append('intg(confidence) for secondary label. specific. until rank 400')
normal_features.append(confidences_secondary_cumsum[:, 399])
adv_features.append(confidences_secondary_cumsum_adv[:, 399])

# guess 12: get delta between first and second confidences (overall).
delta_1st_2nd     = np.empty((test_size, args.num_points), dtype=np.float32)
delta_1st_2nd_adv = np.empty((test_size, args.num_points), dtype=np.float32)

for i in range(test_size):
    for j in range(args.num_points):
        second, first = np.sort(probs[i, j])[-2:]
        delta_1st_2nd[i, j] = first - second
        second, first = np.sort(probs_adv[i, j])[-2:]
        delta_1st_2nd_adv[i, j] = first - second

plt.figure()
plt.plot(delta_1st_2nd[inds[0]], 'blue')
plt.plot(delta_1st_2nd_adv[inds[0]], 'red')
plt.title('delta_1st_2nd overall for i={}'.format(inds[0]))
plt.legend(['normal', 'adv'])
plt.show()

delta_1st_2nd_cumsum     = delta_1st_2nd.cumsum(axis=1)
delta_1st_2nd_cumsum_adv = delta_1st_2nd_adv.cumsum(axis=1)

plt.figure()
plt.hist(delta_1st_2nd_cumsum[f3_inds, 399], alpha=0.5, label='normal', bins=100)#, range=[-1, 40])
plt.hist(delta_1st_2nd_cumsum_adv[f3_inds, 399], alpha=0.5, label='adv', bins=100)#, range=[-1, 40])
plt.legend(loc='upper right')
plt.title('intg(first_label_prob - second_label_prob). overall. until rank 400')
# plt.xlim(-300, 500000)
plt.show()

features_index.append('intg(first_label_prob - second_label_prob). overall. until rank 400')
normal_features.append(delta_1st_2nd_cumsum[:, 399])
adv_features.append(delta_1st_2nd_cumsum_adv[:, 399])

# guess 13: delta between original (primary) label and secondary (first switched) label
delta_1st_2nd_specific     = -1 * np.ones((test_size, args.num_points), dtype=np.float32)
delta_1st_2nd_specific_adv = -1 * np.ones((test_size, args.num_points), dtype=np.float32)

for i in range(test_size):
    for j in range(args.num_points):
        if first_sw_rank[i] != -1:
            delta_1st_2nd_specific[i, j] = confidences_primary[i, j] - confidences_secondary[i, j]
        if first_sw_rank_adv[i] != -1:
            delta_1st_2nd_specific_adv[i, j] = confidences_primary_adv[i, j] - confidences_secondary_adv[i, j]

plt.figure()
plt.plot(delta_1st_2nd_specific[inds[0]], 'blue')
plt.plot(delta_1st_2nd_specific_adv[inds[0]], 'red')
plt.title('delta_1st_2nd specific for i={}'.format(inds[0]))
plt.legend(['normal', 'adv'])
plt.show()

delta_1st_2nd_specific_cumsum     = delta_1st_2nd_specific.cumsum(axis=1)
delta_1st_2nd_specific_cumsum_adv = delta_1st_2nd_specific_adv.cumsum(axis=1)

for i in range(test_size):
    if first_sw_rank[i] == -1:
        delta_1st_2nd_specific_cumsum[i] = np.arange(args.num_points)
    if first_sw_rank_adv[i] == -1:
        delta_1st_2nd_specific_cumsum_adv[i] = np.arange(args.num_points)

plt.figure()
plt.hist(delta_1st_2nd_specific_cumsum[f3_inds, 499], alpha=0.5, label='normal', bins=100)#, range=[0, 300])
plt.hist(delta_1st_2nd_specific_cumsum_adv[f3_inds, 499], alpha=0.5, label='adv', bins=100)#, range=[0, 300])
plt.legend(loc='upper right')
plt.title('intg(primary_prob - secondary_prob). specific. until rank 500')
#plt.ylim(0, 500)
plt.show()

features_index.append('intg(primary_prob - secondary_prob). specific. until rank 500')
normal_features.append(delta_1st_2nd_specific_cumsum[:, 499])
adv_features.append(delta_1st_2nd_specific_cumsum_adv[:, 499])

# stacking features to numpy
normal_features = np.stack(normal_features, axis=1)
adv_features    = np.stack(adv_features, axis=1)
# normal_features = normal_features.reshape((test_size, -1))
# adv_features = adv_features.reshape((test_size, -1))

# using simple threshold, for a single feature_ind
feature_ind = 13
plt.figure()
plt.hist(normal_features[f3_inds_val, feature_ind], alpha=0.5, label='normal', bins=100)#, range=[150, 400])
plt.hist(adv_features[f3_inds_val, feature_ind], alpha=0.5, label='adv', bins=100)#, range=[150, 400])
plt.legend(loc='upper right')
plt.title('hist for {}'.format(features_index[feature_ind]))
plt.ylim(0, 200)
plt.show()

detection_preds     = normal_features[:, feature_ind] < 200
detection_preds_adv = adv_features[:, feature_ind] < 200

# define complete training/testing set for learned models:
train_features = np.concatenate((normal_features[f3_inds_val], adv_features[f3_inds_val]))
train_labels   = np.concatenate((np.zeros(len(f3_inds_val)), np.ones(len(f3_inds_val))))
test_normal_features = normal_features.copy()
test_adv_features = adv_features.copy()

# fitting random forest classifier
clf = RandomForestClassifier(
    n_estimators=1000,
    criterion="entropy",  # gini or entropy
    max_depth=None, # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
                    # until all leaves contain less than min_samples_split samples.
    min_samples_split=10,
    min_samples_leaf=10,
    bootstrap=True, # Whether bootstrap samples are used when building trees.
                    # If False, the whole datset is used to build each tree.
    random_state=rand_gen,
    verbose=1000,
    class_weight={0: 1, 1: 10}
)

clf.fit(train_features, train_labels)
detection_preds     = clf.predict(test_normal_features)
detection_preds_adv = clf.predict(test_adv_features)

# fitting SVM
# clf = LinearSVC(penalty='l2', loss='hinge', verbose=1, random_state=rand_gen, max_iter=100000)
# clf.fit(train_features, train_labels)
# detection_preds     = clf.predict(test_normal_features)
# detection_preds_adv = clf.predict(test_adv_features)

acc_all = np.mean(detection_preds[test_inds] == 0)
acc_f1 = np.mean(detection_preds[f1_inds_test] == 0)
acc_f2 = np.mean(detection_preds[f2_inds_test] == 0)
acc_f3 = np.mean(detection_preds[f3_inds_test] == 0)

acc_all_adv = np.mean(detection_preds_adv[test_inds] == 1)
acc_f1_adv = np.mean(detection_preds_adv[f1_inds_test] == 1)
acc_f2_adv = np.mean(detection_preds_adv[f2_inds_test] == 1)
acc_f3_adv = np.mean(detection_preds_adv[f3_inds_test] == 1)

print('Accuracy for all samples: {:.2f}/{:.2f}%. '
      'f1 samples: {:.2f}/{:.2f}%, f2 samples: {:.2f}/{:.2f}%, f3 samples: {:.2f}/{:.2f}%'
      .format(acc_all * 100, acc_all_adv * 100, acc_f1 * 100, acc_f1_adv * 100, acc_f2 * 100, acc_f2_adv * 100, acc_f3 * 100, acc_f3_adv * 100))

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
from datetime import datetime
from sklearn.decomposition import PCA

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from active_learning_project.attacks.ball_explorer import BallExplorer

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
parser.add_argument('--eps', default=0.01, type=float, help='the ball radius for exploration')
parser.add_argument('--num_points', default=20, type=int, help='the number of gradients to sample')
parser.add_argument('--output', default='loss', type=str, help='pred or loss')

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
    output=args.output
)

print('calculating original gradients in ball...')
x_ball, outputs, grads_norm = explorer.generate(X_test)
print('calculating adv gradients in ball...')
x_ball_adv, outputs_adv, grads_adv = explorer.generate(X_test_adv)
print('done')

# get loss in ball
# first, for each image sort all the samples by L2 distance from the main
x_dist     = np.empty((test_size, args.num_points), dtype=np.float32)
x_adv_dist = np.empty((test_size, args.num_points), dtype=np.float32)
for j in range(args.num_points):
    x_dist[:, j]     = np.linalg.norm((x_ball[:, j] - X_test).reshape((test_size, -1)), axis=1)
    x_adv_dist[:, j] = np.linalg.norm((x_ball_adv[:, j] - X_test_adv).reshape((test_size, -1)), axis=1)
ranks = x_dist.argsort(axis=1)
ranks_adv = x_adv_dist.argsort(axis=1)

i=70
plt.figure()
plt.plot(outputs[i, ranks[i, :]])
plt.title('Loss for x in ball vs L2 distance from original x. i={}'.format(i))
plt.show()

plt.figure()
plt.plot(outputs_adv[i, ranks_adv[i, :]])
plt.title('Loss for x in ball vs L2 distance from original x_adv. i={}'.format(i))
plt.show()





# get stats
grads_norm_mean_abs             = np.abs(grads_norm).mean(axis=(1, 2, 3, 4))
grads_norm_mean_abs_center      = np.abs(grads_norm)[:, 0].mean(axis=(1, 2, 3))
grads_norm_diff_center          = grads_norm[:, 1:] - np.expand_dims(grads_norm[:, 0], axis=1)
grads_norm_diff_center_mean_abs = np.abs(grads_norm_diff_center).mean(axis=(1, 2, 3, 4))

grads_adv_mean_abs              = np.abs(grads_adv).mean(axis=(1, 2, 3, 4))
grads_adv_mean_abs_center       = np.abs(grads_adv)[:, 0].mean(axis=(1, 2, 3))
grads_adv_diff_center           = grads_adv[:, 1:] - np.expand_dims(grads_adv[:, 0], axis=1)
grads_adv_diff_center_mean_abs  = np.abs(grads_adv_diff_center).mean(axis=(1, 2, 3, 4))

# calculate ratio of adv/norm
grads_adv_norm_ratio = grads_adv_diff_center_mean_abs / grads_norm_diff_center_mean_abs
acc = np.mean(grads_adv_norm_ratio[f3_inds] > 1.0)
grads_adv_norm_ratio_just_center = grads_adv_mean_abs_center / grads_norm_mean_abs_center
acc_just_center = np.mean(grads_adv_norm_ratio_just_center[f3_inds] > 1.0)
print('accuracy of adversarial detection. plain: {:.2f}%, with ball: {:.2f}%'.format(100. * acc_just_center, 100. * acc))



# printing PCA for one sample (from f3 inds).
i = 40
plt.close()
plt.figure()
pca = PCA(n_components=2, random_state=rand_gen, whiten=False)
pca.fit(np.abs(grads_norm)[i].reshape(args.num_points, -1))
grads_norm_proj = pca.transform(np.abs(grads_norm)[i].reshape(args.num_points, -1))
plt.scatter(grads_norm_proj[1:, 0], grads_norm_proj[1:, 1], s=10, marker='o', c='blue', label='in ball')
plt.scatter(grads_norm_proj[0, 0], grads_norm_proj[0, 1], s=50, marker='X', c='red', label='original')
x_min, x_max = grads_norm_proj[:, 0].min(), grads_norm_proj[:, 0].max()
y_min, y_max = grads_norm_proj[:, 1].min(), grads_norm_proj[:, 1].max()
x_gap = x_max - x_min
y_gap = y_max - y_min
plt.xlim([x_min - 0.04*x_gap, x_max + 0.04*x_gap])
plt.ylim([y_min - 0.04*y_gap, y_max + 0.04*y_gap])
plt.legend()
plt.title('PCA 2D map for normal images. i={}'.format(i))

plt.figure()
pca = PCA(n_components=2, random_state=rand_gen, whiten=False)
pca.fit(np.abs(grads_adv)[i].reshape(args.num_points, -1))
grads_adv_proj = pca.transform(np.abs(grads_adv)[i].reshape(args.num_points, -1))
plt.scatter(grads_adv_proj[1:, 0], grads_adv_proj[1:, 1], s=10, marker='o', c='blue', label='in ball')
plt.scatter(grads_adv_proj[0, 0], grads_adv_proj[0, 1], s=50, marker='X', c='red', label='original')
x_min, x_max = grads_adv_proj[:, 0].min(), grads_adv_proj[:, 0].max()
y_min, y_max = grads_adv_proj[:, 1].min(), grads_adv_proj[:, 1].max()
x_gap = x_max - x_min
y_gap = y_max - y_min
plt.xlim([x_min - 0.04*x_gap, x_max + 0.04*x_gap])
plt.ylim([y_min - 0.04*y_gap, y_max + 0.04*y_gap])
plt.legend()
plt.title('PCA 2D map for adv images. i={}'.format(i))
plt.show()

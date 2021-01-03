import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import numpy as np
import json
import os
import argparse
import sys
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor
from active_learning_project.attacks.tta_ball_explorer import TTABallExplorer
from active_learning_project.utils import convert_tensor_to_image, calc_prob_wo_l, compute_roc, boolean_string

from active_learning_project.tta_utils import plot_ttas, update_useful_stats, register_intg_loss, \
    register_intg_rel_loss, register_max_rel_loss, register_rank_at_thd_rel_loss, register_rank_at_first_pred_switch, \
    register_num_pred_switches, register_mean_loss_for_initial_label, register_mean_rel_loss_for_initial_label, \
    register_intg_confidences_prime, register_intg_confidences_prime_specific, register_intg_confidences_secondary, \
    register_intg_confidences_secondary_specific, register_intg_delta_confidences_prime_rest, \
    register_intg_delta_confidences_prime_secondary_specific, register_delta_probs_prime_secondary_excl_rest

from active_learning_project.global_vars import features_index, normal_features_list, adv_features_list
import matplotlib.pyplot as plt

from art.classifiers import PyTorchClassifier
from active_learning_project.classifiers.pytorch_ext_classifier import PyTorchExtClassifier

parser = argparse.ArgumentParser(description='PyTorch adversarial robustness testing')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='fgsm_targeted', type=str, help='attack directory')
parser.add_argument('--save_dir', default='tta_ball_rev_L2_eps_2_n_1000', type=str, help='reverse dir')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--subset', default=-1, type=int, help='attack only subset of test set')

# for exploration
parser.add_argument('--collect_normal_ball', default=False, type=boolean_string, help='norm or ball distance')
parser.add_argument('--norm', default='L2', type=str, help='norm or ball distance')
parser.add_argument('--eps', default=2.0, type=float, help='the ball radius for exploration')
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
NORMAL_SAVE_DIR = os.path.join(args.checkpoint_dir, 'normal', args.save_dir)
os.makedirs(os.path.join(NORMAL_SAVE_DIR), exist_ok=True)

SAVE_DIR = os.path.join(ATTACK_DIR, args.save_dir)
os.makedirs(os.path.join(SAVE_DIR, 'inds'), exist_ok=True)

# saving current args:
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

if args.collect_normal_ball and not os.path.exists(os.path.join(NORMAL_SAVE_DIR, 'x_ball_subset_100.npy')):
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

    print('start saving to disk ({})...'.format(NORMAL_SAVE_DIR))
    np.save(os.path.join(NORMAL_SAVE_DIR, 'x_ball_subset_100.npy'), x_ball[0:100])
    np.save(os.path.join(NORMAL_SAVE_DIR, 'losses.npy'), losses)
    np.save(os.path.join(NORMAL_SAVE_DIR, 'preds.npy'), preds)
    np.save(os.path.join(NORMAL_SAVE_DIR, 'noise_powers.npy'), noise_powers)
    with open(os.path.join(NORMAL_SAVE_DIR, 'run_args.txt'), 'w') as f:  # save args to normal dir only once.
        json.dump(args.__dict__, f, indent=2)

    x_ball = x_ball[0:100]  # expensive in memory
else:
    x_ball     = np.load(os.path.join(NORMAL_SAVE_DIR, 'x_ball_subset_100.npy'))
    losses     = np.load(os.path.join(NORMAL_SAVE_DIR, 'losses.npy'))
    preds      = np.load(os.path.join(NORMAL_SAVE_DIR, 'preds.npy'))
    x_dist     = np.load(os.path.join(NORMAL_SAVE_DIR, 'noise_powers.npy'))

if not os.path.exists(os.path.join(SAVE_DIR, 'x_ball_adv_subset_100.npy')):
    print('calculating adv x in ball...')
    x_ball_adv, losses_adv, preds_adv, noise_powers_adv = explorer.generate(X_test_adv)
    print('done calculating x adv ball')

    ranks_adv = noise_powers_adv.argsort(axis=1)

    for i in range(test_size):
        rks_adv = ranks_adv[i]
        x_ball_adv[i]       = x_ball_adv[i, rks_adv]
        losses_adv[i]       = losses_adv[i, rks_adv]
        preds_adv[i]        = preds_adv[i, rks_adv]
        noise_powers_adv[i] = noise_powers_adv[i, rks_adv]

    print('start saving to disk ({})...'.format(SAVE_DIR))
    np.save(os.path.join(SAVE_DIR, 'x_ball_adv_subset_100.npy'), x_ball_adv[0:100])
    np.save(os.path.join(SAVE_DIR, 'losses_adv.npy'), losses_adv)
    np.save(os.path.join(SAVE_DIR, 'preds_adv.npy'), preds_adv)
    np.save(os.path.join(SAVE_DIR, 'noise_powers_adv.npy'), noise_powers_adv)

    x_ball_adv = x_ball_adv[0:100]  # expensive in memory
else:
    x_ball_adv = np.load(os.path.join(SAVE_DIR, 'x_ball_adv_subset_100.npy'))
    losses_adv = np.load(os.path.join(SAVE_DIR, 'losses_adv.npy'))
    preds_adv  = np.load(os.path.join(SAVE_DIR, 'preds_adv.npy'))
    x_dist_adv = np.load(os.path.join(SAVE_DIR, 'noise_powers_adv.npy'))

# converting everything from 3x32x32 to 32x32x3
X_test_img     = convert_tensor_to_image(X_test)
X_test_adv_img = convert_tensor_to_image(X_test_adv)
# x_ball_img     = convert_tensor_to_image(x_ball.reshape((test_size * args.num_points, ) + X_test.shape[1:])) \
#                 .reshape((test_size, args.num_points) + X_test_img.shape[1:])
x_ball_img     = convert_tensor_to_image(x_ball.reshape((100 * args.num_points, ) + X_test.shape[1:])) \
                .reshape((100, args.num_points) + X_test_img.shape[1:])
# x_ball_adv_img = convert_tensor_to_image(x_ball_adv.reshape((test_size * args.num_points, ) + X_test.shape[1:])) \
#                 .reshape((test_size, args.num_points) + X_test_img.shape[1:])
x_ball_adv_img = convert_tensor_to_image(x_ball_adv.reshape((100 * args.num_points, ) + X_test.shape[1:])) \
                .reshape((100, args.num_points) + X_test_img.shape[1:])

assert np.all(X_test_img[0]     == x_ball_img[0, 0]), 'first normal image must match'
assert np.all(X_test_adv_img[0] == x_ball_adv_img[0, 0]), 'first adv image must match'

# visualizing the images in the ball
plot_ttas(x_ball_img, x_ball_adv_img, f2_inds)

# wrap useful stats in a dictionary:
stats = {}
stats['preds'] = preds
stats['losses'] = losses

stats_adv = {}
stats_adv['preds'] = preds_adv
stats_adv['losses'] = losses_adv

print('updating useful stats for normal images...')
update_useful_stats(stats)
print('updating useful stats for adv images...')
update_useful_stats(stats_adv)

assert np.all(stats['y_ball_preds'][:, 0] == y_test_preds)
assert np.all(stats_adv['y_ball_preds'][:, 0] == y_test_adv_preds)

print('calculating Feature 1: integral(loss)...')
register_intg_loss(stats, stats_adv, f2_inds_val)

print('calculating Feature 2: integral(rel_loss)...')
register_intg_rel_loss(stats, stats_adv, f2_inds_val)

print('calculating Feature 3: max(rel_loss)...')
register_max_rel_loss(stats, stats_adv, f2_inds_val)

print('calculating Feature 4: rank @rel_loss= thd * max_rel_loss...')
register_rank_at_thd_rel_loss(stats, stats_adv, f2_inds_val)

print('calculating Feature 5: rank @first pred switch...')
register_rank_at_first_pred_switch(stats, stats_adv, f2_inds_val)

print('calculating Feature 6: number of switches until a specific rank...')
register_num_pred_switches(stats, stats_adv, f2_inds_val)

print('calculating Feature 7: mean(loss) for only initial label...')
register_mean_loss_for_initial_label(stats, stats_adv, f2_inds_val)

print('calculating Feature 8: mean(rel_loss) for only initial label...')
register_mean_rel_loss_for_initial_label(stats, stats_adv, f2_inds_val)

print('calculating Feature 9: integral(confidence) until a specific rank...')
register_intg_confidences_prime(stats, stats_adv, f2_inds_val)

print('calculating Feature 10: integral(confidence) for primary only. Specific...')
register_intg_confidences_prime_specific(stats, stats_adv, f2_inds_val)

print('calculating Feature 11: integral(confidence) for secondary label. Overall...')
register_intg_confidences_secondary(stats, stats_adv, f2_inds_val)

print('calculating Feature 12: integral(confidence) for secondary label. Specific...')
register_intg_confidences_secondary_specific(stats, stats_adv, f2_inds_val)

print('calculating Feature 13: integral(delta) for prime - rest. Overall...')
register_intg_delta_confidences_prime_rest(stats, stats_adv, f2_inds_val)

print('calculating Feature 14: integral(delta) for prime - secondary. Specific...')
register_intg_delta_confidences_prime_secondary_specific(stats, stats_adv, f2_inds_val)

print('calculating Feature 15: integral(delta) for prime - secondary, setting zj=-inf for other labels...')
register_delta_probs_prime_secondary_excl_rest(stats, stats_adv, f2_inds_val)

# debug - get all ranks
# with open(os.path.join(SAVE_DIR, 'features_index_hist_all.pkl'), 'wb') as f:
#     pickle.dump(features_index, f)
# with open(os.path.join(SAVE_DIR, 'normal_features_list_all.pkl'), 'wb') as f:
#     pickle.dump(normal_features_list, f)
# with open(os.path.join(SAVE_DIR, 'adv_features_hist_all.pkl'), 'wb') as f:
#     pickle.dump(adv_features_list, f)

# feature_ind = 0
# plt.figure()
# plt.hist(normal_features_list[feature_ind][f2_inds_val], alpha=0.5, label='normal', bins=100)#, range=[150, 400])
# plt.hist(adv_features[f2_inds_val, feature_ind], alpha=0.5, label='adv', bins=100)#, range=[150, 400])
# plt.legend(loc='upper right')
# plt.title('hist for {}'.format(features_index[feature_ind]))
# plt.ylim(0, 200)
# plt.show()

# stacking features to numpy
features_index = np.asarray(features_index)
normal_features = np.stack(normal_features_list, axis=1)
adv_features    = np.stack(adv_features_list, axis=1)
# normal_features = normal_features.reshape((test_size, -1))
# adv_features = adv_features.reshape((test_size, -1))

# saving everything
np.save(os.path.join(SAVE_DIR, 'features_index_hist.npy'), features_index)
np.save(os.path.join(SAVE_DIR, 'normal_features_hist.npy'), normal_features)
np.save(os.path.join(SAVE_DIR, 'adv_features_hist.npy'), adv_features)

np.save(os.path.join(SAVE_DIR, 'inds', 'val_inds.npy'), val_inds)
np.save(os.path.join(SAVE_DIR, 'inds', 'f0_inds_val.npy'), f0_inds_val)
np.save(os.path.join(SAVE_DIR, 'inds', 'f1_inds_val.npy'), f1_inds_val)
np.save(os.path.join(SAVE_DIR, 'inds', 'f2_inds_val.npy'), f2_inds_val)
np.save(os.path.join(SAVE_DIR, 'inds', 'f3_inds_val.npy'), f3_inds_val)

np.save(os.path.join(SAVE_DIR, 'inds', 'test_inds.npy'), test_inds)
np.save(os.path.join(SAVE_DIR, 'inds', 'f0_inds_test.npy'), f0_inds_test)
np.save(os.path.join(SAVE_DIR, 'inds', 'f1_inds_test.npy'), f1_inds_test)
np.save(os.path.join(SAVE_DIR, 'inds', 'f2_inds_test.npy'), f2_inds_test)
np.save(os.path.join(SAVE_DIR, 'inds', 'f3_inds_test.npy'), f3_inds_test)

print('done')
exit(0)

# define complete training/testing set for learned models:
train_features = np.concatenate((normal_features[f2_inds_val], adv_features[f2_inds_val]))
train_labels   = np.concatenate((np.zeros(len(f2_inds_val)), np.ones(len(f2_inds_val))))
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
    n_jobs=20
    # class_weight={0: 1, 1: 10}
)

clf.fit(train_features, train_labels)
detection_preds     = clf.predict(test_normal_features)
detection_preds_adv = clf.predict(test_adv_features)
detection_preds_prob     = clf.predict_proba(test_normal_features)[:, 1]
detection_preds_prob_adv = clf.predict_proba(test_adv_features)[:, 1]

# fitting SVM
# clf = LinearSVC(penalty='l2', loss='hinge', verbose=1, random_state=rand_gen, max_iter=100000)
# clf.fit(train_features, train_labels)
# detection_preds     = clf.predict(test_normal_features)
# detection_preds_adv = clf.predict(test_adv_features)

# adv detection metrics:
print('Calculating adv detection metrics...')
acc_all = np.mean(detection_preds[test_inds] == 0)
acc_f1 = np.mean(detection_preds[f1_inds_test] == 0)
acc_f2 = np.mean(detection_preds[f2_inds_test] == 0)
acc_f3 = np.mean(detection_preds[f3_inds_test] == 0)

acc_all_adv = np.mean(detection_preds_adv[test_inds] == 1)
acc_f1_adv = np.mean(detection_preds_adv[f1_inds_test] == 1)
acc_f2_adv = np.mean(detection_preds_adv[f2_inds_test] == 1)
acc_f3_adv = np.mean(detection_preds_adv[f3_inds_test] == 1)

f2_test_preds_all = np.concatenate((detection_preds_prob[f2_inds_test], detection_preds_prob_adv[f2_inds_test]), axis=0)
f2_test_gt_all    = np.concatenate((np.zeros(len(f2_inds_test)), np.ones(len(f2_inds_test))), axis=0)
_, _, auc_score = compute_roc(f2_test_gt_all, f2_test_preds_all, plot=True)

print('Accuracy for all samples: {:.2f}/{:.2f}%. '
      'f1 samples: {:.2f}/{:.2f}%, f2 samples: {:.2f}/{:.2f}%, f3 samples: {:.2f}/{:.2f}%. '
      'AUC score: {:.5f}'
      .format(acc_all * 100, acc_all_adv * 100, acc_f1 * 100, acc_f1_adv * 100, acc_f2 * 100, acc_f2_adv * 100,
              acc_f3 * 100, acc_f3_adv * 100, auc_score))

# debugging misclassification
false_positive_inds = [i for i in np.where(detection_preds     == 1)[0] if i in f2_inds_test]
false_negative_inds = [i for i in np.where(detection_preds_adv == 0)[0] if i in f2_inds_test]

print('Calculating robustness metrics...')
# summing up the probs (after softmax) for every class, for every TTA sample
# robustness_preds     = stats['probs_mean'].argmax(axis=1)
# robustness_preds_adv = stats_adv['probs_mean'].argmax(axis=1)

# confining ourself to only the first and second most likely classes in the original image:
# robustness_preds     = np.empty(test_size)
# robustness_preds_adv = np.empty(test_size)
# for k in range(test_size):
#     candidates = stats['probs'][k, 0].argsort()[[-1, -2]]
#     first_score = stats['probs_mean'][k, candidates[0]]
#     second_score = stats['probs_mean'][k, candidates[1]]
#     robustness_preds[k] = candidates[0] if first_score >= second_score else candidates[1]
#
#     candidates = stats_adv['probs'][k, 0].argsort()[[-1, -2]]
#     first_score = stats_adv['probs_mean'][k, candidates[0]]
#     second_score = stats_adv['probs_mean'][k, candidates[1]]
#     robustness_preds_adv[k] = candidates[0] if first_score >= second_score else candidates[1]

# confining ourself to only the first and second most likely classes in the original image, with pil_mat
# robustness_preds     = np.empty(test_size)
# robustness_preds_adv = np.empty(test_size)
# for k in range(test_size):
#     candidates = stats['probs'][k, 0].argsort()[[-1, -2]]
#     first_score = stats['pil_mat_mean'][k, candidates[1], candidates[0]]
#     second_score = stats['pil_mat_mean'][k, candidates[0], candidates[1]]
#     robustness_preds[k] = candidates[0] if first_score >= second_score else candidates[1]
#
#     candidates = stats['probs'][k, 0].argsort()[[-1, -2]]
#     first_score = stats_adv['pil_mat_mean'][k, candidates[1], candidates[0]]
#     second_score = stats_adv['pil_mat_mean'][k, candidates[0], candidates[1]]
#     robustness_preds_adv[k] = candidates[0] if first_score >= second_score else candidates[1]

# converting detection_preds to robustness_preds
robustness_probs     = np.empty((test_size, len(classes)))
robustness_probs_adv = np.empty((test_size, len(classes)))
for k in range(test_size):
    l = stats['y_ball_preds'][k, 0]
    p_is_adv = detection_preds_prob[k]
    p_vec_norm = stats['probs'][k, 0]
    p_vec_adv = calc_prob_wo_l(stats['preds'][k, 0], l=l)
    robustness_probs[k] = p_vec_adv * p_is_adv + p_vec_norm * (1 - p_is_adv)

    l = stats_adv['y_ball_preds'][k, 0]
    p_is_adv = detection_preds_prob_adv[k]
    p_vec_norm = stats_adv['probs'][k, 0]
    p_vec_adv = calc_prob_wo_l(stats_adv['preds'][k, 0], l=l)
    robustness_probs_adv[k] = p_vec_adv * p_is_adv + p_vec_norm * (1 - p_is_adv)

robustness_preds     = robustness_probs.argmax(axis=1)
robustness_preds_adv = robustness_probs_adv.argmax(axis=1)

# try with mean
# robustness_probs     = np.empty((test_size, len(classes)))
# robustness_probs_adv = np.empty((test_size, len(classes)))
# for k in range(test_size):
#     l = stats['y_ball_preds'][k, 0]
#     p_is_adv = detection_preds_prob[k]
#     p_vec_norm = stats['probs'][k].mean(axis=0)
#     p_vec_adv = stats['pil_mat_mean'][k, l]
#     robustness_probs[k] = p_vec_adv * p_is_adv + p_vec_norm * (1 - p_is_adv)
#
#     l = stats_adv['y_ball_preds'][k, 0]
#     p_is_adv = detection_preds_prob_adv[k]
#     p_vec_norm = stats_adv['probs'][k].mean(axis=0)
#     p_vec_adv = stats_adv['pil_mat_mean'][k, l]
#     robustness_probs_adv[k] = p_vec_adv * p_is_adv + p_vec_norm * (1 - p_is_adv)
# robustness_preds     = robustness_probs.argmax(axis=1)
# robustness_preds_adv = robustness_probs_adv.argmax(axis=1)

# robustness metrics
acc_all = np.mean(robustness_preds[test_inds] == y_test[test_inds])
acc_f1 = np.mean(robustness_preds[f1_inds_test] == y_test[f1_inds_test])
acc_f2 = np.mean(robustness_preds[f2_inds_test] == y_test[f2_inds_test])
acc_f3 = np.mean(robustness_preds[f3_inds_test] == y_test[f3_inds_test])

acc_all_adv = np.mean(robustness_preds_adv[test_inds] == y_test[test_inds])
acc_f1_adv = np.mean(robustness_preds_adv[f1_inds_test] == y_test[f1_inds_test])
acc_f2_adv = np.mean(robustness_preds_adv[f2_inds_test] == y_test[f2_inds_test])
acc_f3_adv = np.mean(robustness_preds_adv[f3_inds_test] == y_test[f3_inds_test])

print('Accuracy: all samples: {:.2f}/{:.2f}%, f1 samples: {:.2f}/{:.2f}%, f2 samples: {:.2f}/{:.2f}%, f3 samples: {:.2f}/{:.2f}%'
      .format(acc_all * 100, acc_all_adv * 100, acc_f1 * 100, acc_f1_adv * 100, acc_f2 * 100, acc_f2_adv * 100, acc_f3 * 100, acc_f3_adv * 100))

# debugging robustness:
# plotting probabilities mean values
# for each image, and for each i, calculate for l (first pred) the exp(zi)/exp(zj) for each j!=l.
probs_wo_l     = np.zeros((test_size, args.num_points, len(classes)))
probs_wo_l_adv = np.zeros((test_size, args.num_points, len(classes)))
for k in range(test_size):
    l = preds[k, 0].argmax()
    for n in range(args.num_points):
        probs_wo_l[k, n] = calc_prob_wo_l(preds[k, n], l)

    l = preds_adv[k, 0].argmax()
    for n in range(args.num_points):
         probs_wo_l_adv[k, n] = calc_prob_wo_l(preds_adv[k, n], l)

probs_wo_l_mean     = probs_wo_l.mean(axis=1)
probs_wo_l_mean_adv = probs_wo_l_adv.mean(axis=1)

confidences_wo_l     = np.max(probs_wo_l, axis=2)
confidences_wo_l_adv = np.max(probs_wo_l_adv, axis=2)

plt.close('all')
i = 125
n_cols = len(classes)
n_rows = 4
fig = plt.figure(figsize=(n_cols, 4 * n_rows))
loc = 1
color_vec = ['blue'] * len(classes)
color_vec[y_test[i]] = 'green'
ax1 = fig.add_subplot(n_rows, 1, loc)
ax1.set_xticklabels(classes, fontdict={'size': 12})
ax1.set_xticks(np.arange(len(classes)))
ax1.set_ylabel('normal <probs>', color='blue', fontdict={'size': 12})
ax1.set_ylim(-0.05, 1.05)
ax1.bar(np.arange(len(classes)), stats['probs_mean'][i], color=color_vec)
ax1.tick_params(axis='y', labelcolor='blue')

loc = 2
color_vec = ['blue'] * len(classes)
color_vec[y_test[i]] = 'green'
ax2 = fig.add_subplot(n_rows, 1, loc)
ax2.set_xticklabels(classes, fontdict={'size': 12})
ax2.set_xticks(np.arange(len(classes)))
ax2.set_ylabel('normal <probs_wo_l>', color='blue', fontdict={'size': 12})
ax2.set_ylim(-0.05, 1.05)
ax2.bar(np.arange(len(classes)), probs_wo_l_mean[i], color=color_vec)
ax2.tick_params(axis='y', labelcolor='blue')

loc = 3
color_vec = ['blue'] * len(classes)
color_vec[y_test[i]] = 'green'
color_vec[stats['y_ball_preds'][i, 0]] = 'red'
ax3 = fig.add_subplot(n_rows, 1, loc)
ax3.set_xticklabels(classes, fontdict={'size': 12})
ax3.set_xticks(np.arange(len(classes)))
ax3.set_ylabel('adv <probs>', color='red', fontdict={'size': 12})
ax3.set_ylim(-0.05, 1.05)
ax3.bar(np.arange(len(classes)), stats_adv['probs_mean'][i], color=color_vec)
ax3.tick_params(axis='y', labelcolor='blue')

loc = 4
color_vec = ['blue'] * len(classes)
color_vec[y_test[i]] = 'green'
color_vec[stats_adv['y_ball_preds'][i, 0]] = 'red'
ax4 = fig.add_subplot(n_rows, 1, loc)
ax4.set_xticklabels(classes, fontdict={'size': 12})
ax4.set_xticks(np.arange(len(classes)))
ax4.set_ylabel('adv <probs_wo_l>', color='red', fontdict={'size': 12})
ax4.set_ylim(-0.05, 1.05)
ax4.bar(np.arange(len(classes)), probs_wo_l_mean_adv[i], color=color_vec)
ax4.tick_params(axis='y', labelcolor='blue')

plt.tight_layout()
plt.show()

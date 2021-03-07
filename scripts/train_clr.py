"""
This script loads a pretrained checkpoint and used the base model to extract features (h), which on top of them we
calculate the contrastive loss using a simple MLP g(h).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import time
import sys
import PIL
import scipy

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from art.estimators.classification import PyTorchClassifier

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.models.projection_head import ProjectobHead
import active_learning_project.datasets.my_transforms as my_transforms

from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_all_data_loader, get_normalized_tensor, get_single_img_dataloader
from active_learning_project.utils import remove_substr_from_keys, boolean_string, save_features, pytorch_evaluate, \
    majority_vote
from torchsummary import summary
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch CLR training on base pretrained net')
parser.add_argument('--checkpoint_dir',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00',
                    type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='cw_targeted', type=str, help='attack directory')

# train
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--steps', default=7, type=int, help='number of training steps')
parser.add_argument('--batch_size', default=16, type=int, help='batch size for the CLR training')
parser.add_argument('--opt', default='adam', type=str, help='optimizer')
parser.add_argument('--mom', default=0.0, type=float, help='momentum of optimizer')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
parser.add_argument('--lambda_ent', default=0.0, type=float, help='Regularization for entropy loss')

# eval
parser.add_argument('--tta_size', default=50, type=int, help='number of test-time augmentations in eval phase')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

# debug
NUM_DEBUG_SAMPLES = 500
TRAIN_TIME_CNT = 0.0
TEST_TIME_CNT = 0.0

def calc_robust_metrics(robustness_preds, robustness_preds_adv):
    if NUM_DEBUG_SAMPLES is not None:
        acc_all = np.mean(robustness_preds[0:NUM_DEBUG_SAMPLES] == y_test[0:NUM_DEBUG_SAMPLES])
        acc_all_adv = np.mean(robustness_preds_adv[0:NUM_DEBUG_SAMPLES] == y_test[0:NUM_DEBUG_SAMPLES])
        print('Robust classification accuracy: all samples: {:.2f}/{:.2f}%'.format(acc_all * 100, acc_all_adv * 100))
    else:
        acc_all = np.mean(robustness_preds[test_inds] == y_test[test_inds])
        acc_f1 = np.mean(robustness_preds[f1_inds_test] == y_test[f1_inds_test])
        acc_f2 = np.mean(robustness_preds[f2_inds_test] == y_test[f2_inds_test])
        acc_f3 = np.mean(robustness_preds[f3_inds_test] == y_test[f3_inds_test])

        acc_all_adv = np.mean(robustness_preds_adv[test_inds] == y_test[test_inds])
        acc_f1_adv = np.mean(robustness_preds_adv[f1_inds_test] == y_test[f1_inds_test])
        acc_f2_adv = np.mean(robustness_preds_adv[f2_inds_test] == y_test[f2_inds_test])
        acc_f3_adv = np.mean(robustness_preds_adv[f3_inds_test] == y_test[f3_inds_test])

        print('Robust classification accuracy: all samples: {:.2f}/{:.2f}%, f1 samples: {:.2f}/{:.2f}%, f2 samples: {:.2f}/{:.2f}%, f3 samples: {:.2f}/{:.2f}%'
              .format(acc_all * 100, acc_all_adv * 100, acc_f1 * 100, acc_f1_adv * 100, acc_f2 * 100, acc_f2_adv * 100, acc_f3 * 100, acc_f3_adv * 100))

def calc_robust_metrics_from_probs_majority_vote(tta_robustness_probs, tta_robustness_probs_adv):
    print('Calculating robustness metrics from probs via majority vote...')
    tta_robustness_preds = tta_robustness_probs.argmax(axis=2)
    robustness_preds = np.apply_along_axis(majority_vote, axis=1, arr=tta_robustness_preds)
    tta_robustness_preds_adv = tta_robustness_probs_adv.argmax(axis=2)
    robustness_preds_adv = np.apply_along_axis(majority_vote, axis=1, arr=tta_robustness_preds_adv)
    calc_robust_metrics(robustness_preds, robustness_preds_adv)

def calc_robust_metrics_from_probs_summation(tta_robustness_probs, tta_robustness_probs_adv):
    print('Calculating robustness metrics from probs via summation...')
    tta_robustness_probs_sum = tta_robustness_probs.sum(axis=1)
    robustness_preds = tta_robustness_probs_sum.argmax(axis=1)
    tta_robustness_probs_adv_sum = tta_robustness_probs_adv.sum(axis=1)
    robustness_preds_adv = tta_robustness_probs_adv_sum.argmax(axis=1)
    calc_robust_metrics(robustness_preds, robustness_preds_adv)

def get_preds_from_emb_center(tta_embedding):
    tta_embeddings_center = tta_embedding.mean(axis=0)
    pred = net.linear(tta_embeddings_center).argmax().detach().cpu().numpy()
    return pred

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
args.dataset = train_args['dataset']

ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

rand_gen = np.random.RandomState(12345)
batch_size = args.batch_size

# get info about attack success:
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

# Data
print('==> Preparing data..')
testloader = get_test_loader(
    dataset=train_args['dataset'],
    batch_size=100,
    num_workers=1,
    pin_memory=True
)

p_hflip = 0.5 if 'cifar' in args.dataset else 0.0
tta_transforms = transforms.Compose([
    my_transforms.ColorJitterPro(
        brightness=[0.6, 1.4],
        contrast=[0.7, 1.3],
        saturation=[0.5, 1.5],
        hue=[-0.06, 0.06],
        gamma=[0.7, 1.3]
    ),
    transforms.Pad(padding=16, padding_mode='edge'),
    transforms.RandomAffine(
        degrees=[-15, 15],
        translate=(4.0 / 64, 4.0 / 64),
        scale=(0.9, 1.1),
        shear=None,
        resample=PIL.Image.BILINEAR,
        fillcolor=None
    ),
    transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.5]),
    transforms.CenterCrop(size=32),
    transforms.RandomHorizontalFlip(p=p_hflip),
])

global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))

classes = testloader.dataset.classes
X_test           = get_normalized_tensor(testloader)
y_test           = np.asarray(testloader.dataset.targets)
X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
if targeted:
    y_test_adv   = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))
y_test_preds     = np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy'))
y_test_adv_preds = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))
test_size = len(X_test)

# Model
if train_args['net'] == 'resnet34':
    net = ResNet34(num_classes=len(classes), activation=train_args['activation'])
elif train_args['net'] == 'resnet101':
    net = ResNet101(num_classes=len(classes), activation=train_args['activation'])
else:
    raise AssertionError("network {} is unknown".format(train_args['net']))
net = net.to(device)
proj_head = ProjectobHead(512, 512, 128)
proj_head = proj_head.to(device)

# summary(net, (3, 32, 32))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
train_params = list(net.parameters()) + list(proj_head.parameters())

def reset_net():
    net.load_state_dict(global_state['best_net'])

def reset_proj():
    for layer in proj_head.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def reset_opt():
    global optimizer
    if args.opt == 'sgd':
        optimizer = optim.SGD(
            train_params,
            lr=args.lr,
            momentum=args.mom,
            weight_decay=args.wd,
            nesterov=args.mom > 0)
    elif args.opt == 'adam':
        optimizer = optim.Adam(
            train_params,
            lr=args.lr,
            weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(
            train_params,
            lr=args.lr,
            weight_decay=args.wd)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            train_params,
            lr=args.lr,
            momentum=args.mom,
            weight_decay=args.wd)
    else:
        raise AssertionError('optimizer {} is not expected'.format(args.opt))

def contrastive_loss(hidden, temperature=0.1):
    hidden1 = hidden[0:args.batch_size]
    hidden2 = hidden[args.batch_size:]
    hidden1 = nn.functional.normalize(hidden1)
    hidden2 = nn.functional.normalize(hidden2)
    cosine_sim = torch.matmul(hidden1, hidden2.T).sum()
    return -cosine_sim

def entropy_loss(logits):
    b = F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)
    b = -1.0 * b.sum()
    return b

robustness_preds     = -1 * np.ones(test_size, dtype=np.int32)
robustness_preds_adv = -1 * np.ones(test_size, dtype=np.int32)
# multi TTAs
robustness_probs     = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)
robustness_probs_adv = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)
embeddings_arr       = -1 * torch.ones((test_size, args.tta_size, net.linear.weight.shape[1]),
                                       dtype=torch.float32, device=device, requires_grad=False)
embeddings_arr_adv   = -1 * torch.ones((test_size, args.tta_size, net.linear.weight.shape[1]),
                                       dtype=torch.float32, device=device, requires_grad=False)
robustness_preds_from_emb_enter     = -1 * np.ones(test_size, dtype=np.int32)
robustness_preds_from_emb_enter_adv = -1 * np.ones(test_size, dtype=np.int32)

reset_opt()
classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=contrastive_loss,
                               optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(classes))

img_cnt = NUM_DEBUG_SAMPLES if NUM_DEBUG_SAMPLES is not None else test_size
for img_ind in tqdm(range(img_cnt)):
    # debug: run only 100 pics
    # debug: only one pic
    # img_ind = 0

    # for normal:
    # Training
    start_time = time.time()
    reset_net()
    reset_proj()
    reset_opt()
    net.eval()
    proj_head.eval()
    train_loader = get_single_img_dataloader(args.dataset, X_test, y_test_preds, 2 * args.batch_size,
                                             pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    for step in range(args.steps):
        (inputs, targets) = list(train_loader)[0]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = net(inputs)
        embeddings, logits = out['embeddings'], out['logits']
        z = proj_head(embeddings)
        loss_cont = contrastive_loss(z)
        loss_ent = entropy_loss(logits)
        loss = loss_cont + args.lambda_ent * loss_ent
        loss.backward()
        optimizer.step()
    TRAIN_TIME_CNT += time.time() - start_time

    # Evaluating
    start_time = time.time()
    robustness_preds[img_ind] = classifier.predict(np.expand_dims(X_test[img_ind], 0)).squeeze().argmax()
    with torch.no_grad():
        tta_cnt = 0
        while tta_cnt < args.tta_size:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                b = tta_cnt
                e = min(tta_cnt + len(inputs), args.tta_size)
                out = net(inputs)
                embeddings_arr[img_ind, b:e] = out['embeddings'][0:(e-b)]
                robustness_probs[img_ind, b:e] = out['probs'][0:(e-b)].detach().cpu().numpy()
                tta_cnt += e-b
                assert tta_cnt <= args.tta_size, 'not cool!'

        robustness_preds_from_emb_enter[img_ind] = get_preds_from_emb_center(embeddings_arr[img_ind])
    TEST_TIME_CNT += time.time() - start_time

    # for adv:
    # Training
    start_time = time.time()
    reset_net()
    reset_proj()
    reset_opt()
    net.eval()
    proj_head.eval()
    train_loader = get_single_img_dataloader(args.dataset, X_test_adv, y_test_adv_preds, 2 * args.batch_size,
                                             pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    for step in range(args.steps):
        (inputs, targets) = list(train_loader)[0]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = net(inputs)
        embeddings, logits = out['embeddings'], out['logits']
        z = proj_head(embeddings)
        loss_cont = contrastive_loss(z)
        loss_ent = entropy_loss(logits)
        loss = loss_cont + args.lambda_ent * loss_ent
        loss.backward()
        optimizer.step()
    TRAIN_TIME_CNT += time.time() - start_time

    # Evaluating
    start_time = time.time()
    robustness_preds_adv[img_ind] = classifier.predict(np.expand_dims(X_test_adv[img_ind], 0)).squeeze().argmax()
    with torch.no_grad():
        tta_cnt = 0
        while tta_cnt < args.tta_size:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                b = tta_cnt
                e = min(tta_cnt + len(inputs), args.tta_size)
                out = net(inputs)
                embeddings_arr_adv[img_ind, b:e] = out['embeddings'][0:(e-b)]
                robustness_probs_adv[img_ind, b:e] = out['probs'][0:(e-b)].detach().cpu().numpy()
                tta_cnt += e-b
                assert tta_cnt <= args.tta_size, 'not cool!'

        robustness_preds_from_emb_enter_adv[img_ind] = get_preds_from_emb_center(embeddings_arr_adv[img_ind])
    TEST_TIME_CNT += time.time() - start_time

if NUM_DEBUG_SAMPLES is not None:
    average_train_time = TRAIN_TIME_CNT / (2 * NUM_DEBUG_SAMPLES)
    average_test_time = TEST_TIME_CNT / (2 * NUM_DEBUG_SAMPLES)
else:
    average_train_time = TRAIN_TIME_CNT / (2 * test_size)
    average_test_time = TEST_TIME_CNT / (2 * test_size)
print('average train/test time per sample: {}/{} secs'.format(average_train_time, average_test_time))

np.save(os.path.join(ATTACK_DIR, 'robustness_preds.npy'), robustness_preds)
np.save(os.path.join(ATTACK_DIR, 'robustness_preds_adv.npy'), robustness_preds_adv)
np.save(os.path.join(ATTACK_DIR, 'robustness_probs.npy'), robustness_probs)
np.save(os.path.join(ATTACK_DIR, 'robustness_probs_adv.npy'), robustness_probs_adv)
np.save(os.path.join(ATTACK_DIR, 'robustness_preds_from_emb_enter.npy'), robustness_preds_from_emb_enter)
np.save(os.path.join(ATTACK_DIR, 'robustness_preds_from_emb_enter_adv.npy'), robustness_preds_from_emb_enter_adv)

print('Calculating robustness metrics...')
calc_robust_metrics(robustness_preds, robustness_preds_adv)
calc_robust_metrics_from_probs_majority_vote(robustness_probs, robustness_probs_adv)
calc_robust_metrics_from_probs_summation(robustness_probs, robustness_probs_adv)
print('Calculating robustness metrics via embedding center...')
calc_robust_metrics(robustness_preds_from_emb_enter, robustness_preds_from_emb_enter_adv)

print('done')

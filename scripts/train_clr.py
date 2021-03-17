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
import logging
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import time
import sys
import PIL
import scipy
import copy
import pickle
from datetime import datetime

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

eps = 1e-8

parser = argparse.ArgumentParser(description='PyTorch CLR training on base pretrained net')
parser.add_argument('--checkpoint_dir',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00',
                    type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='cw_targeted', type=str, help='attack directory')
parser.add_argument('--dump_dir', default='dump', type=str, help='attack directory')

# train
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--steps_inc_ent', default=4, type=int, help='number of pre-training steps to decrease sim')
parser.add_argument('--steps', default=15, type=int, help='number of training steps')
parser.add_argument('--batch_size', default=32, type=int, help='batch size for the CLR training')
parser.add_argument('--opt', default='adam', type=str, help='optimizer: sgd, adam, rmsprop')
parser.add_argument('--mom', default=0.0, type=float, help='momentum of optimizer')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
parser.add_argument('--lambda_ent', default=0.0, type=float, help='Regularization for entropy loss')
parser.add_argument('--lambda_wdiff', default=100.0, type=float, help='Regularization for weight diff')

# eval
parser.add_argument('--tta_size', default=50, type=int, help='number of test-time augmentations in eval phase')

# debug:
parser.add_argument('--debug_size', default=None, type=int, help='number of image to run in debug mode')
parser.add_argument('--dump', '-d', action='store_true', help='resume from checkpoint')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

# debug
NUM_DEBUG_SAMPLES = args.debug_size
TRAIN_TIME_CNT = 0.0
TEST_TIME_CNT = 0.0
ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
DUMP_DIR = os.path.join(ATTACK_DIR, args.dump_dir)
os.makedirs(DUMP_DIR, exist_ok=True)

logging.basicConfig(filename=os.path.join(DUMP_DIR, 'log.log'),
                    filemode='w',
                    format='%(asctime)s %(name)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

def log(str):
    logging.info(str)
    print(str)

def calc_robust_metrics(robustness_preds, robustness_preds_adv):
    if NUM_DEBUG_SAMPLES is not None:
        acc_all = np.mean(robustness_preds[0:NUM_DEBUG_SAMPLES] == y_test[0:NUM_DEBUG_SAMPLES])
        acc_all_adv = np.mean(robustness_preds_adv[0:NUM_DEBUG_SAMPLES] == y_test[0:NUM_DEBUG_SAMPLES])
        log('Robust classification accuracy: all samples: {:.2f}/{:.2f}%'.format(acc_all * 100, acc_all_adv * 100))
    else:
        acc_all = np.mean(robustness_preds[test_inds] == y_test[test_inds])
        acc_f1 = np.mean(robustness_preds[f1_inds_test] == y_test[f1_inds_test])
        acc_f2 = np.mean(robustness_preds[f2_inds_test] == y_test[f2_inds_test])
        acc_f3 = np.mean(robustness_preds[f3_inds_test] == y_test[f3_inds_test])

        acc_all_adv = np.mean(robustness_preds_adv[test_inds] == y_test[test_inds])
        acc_f1_adv = np.mean(robustness_preds_adv[f1_inds_test] == y_test[f1_inds_test])
        acc_f2_adv = np.mean(robustness_preds_adv[f2_inds_test] == y_test[f2_inds_test])
        acc_f3_adv = np.mean(robustness_preds_adv[f3_inds_test] == y_test[f3_inds_test])

        log('Robust classification accuracy: all samples: {:.2f}/{:.2f}%, f1 samples: {:.2f}/{:.2f}%, f2 samples: {:.2f}/{:.2f}%, f3 samples: {:.2f}/{:.2f}%'
            .format(acc_all * 100, acc_all_adv * 100, acc_f1 * 100, acc_f1_adv * 100, acc_f2 * 100, acc_f2_adv * 100, acc_f3 * 100, acc_f3_adv * 100))

def calc_robust_metrics_from_probs_majority_vote(tta_robustness_probs, tta_robustness_probs_adv):
    log('Calculating robustness metrics from probs via majority vote...')
    tta_robustness_preds = tta_robustness_probs.argmax(axis=2)
    robustness_preds = np.apply_along_axis(majority_vote, axis=1, arr=tta_robustness_preds)
    tta_robustness_preds_adv = tta_robustness_probs_adv.argmax(axis=2)
    robustness_preds_adv = np.apply_along_axis(majority_vote, axis=1, arr=tta_robustness_preds_adv)
    calc_robust_metrics(robustness_preds, robustness_preds_adv)

def calc_robust_metrics_from_probs_summation(tta_robustness_probs, tta_robustness_probs_adv):
    log('Calculating robustness metrics from probs via summation...')
    tta_robustness_probs_sum = tta_robustness_probs.sum(axis=1)
    robustness_preds = tta_robustness_probs_sum.argmax(axis=1)
    tta_robustness_probs_adv_sum = tta_robustness_probs_adv.sum(axis=1)
    robustness_preds_adv = tta_robustness_probs_adv_sum.argmax(axis=1)
    calc_robust_metrics(robustness_preds, robustness_preds_adv)

def get_probs_from_emb_center(tta_embedding):
    tta_embeddings_center = tta_embedding.mean(axis=0)
    probs = F.softmax(net.linear(tta_embeddings_center))
    return probs

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
args.dataset = train_args['dataset']

with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

rand_gen = np.random.RandomState(12345)
normal_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'normal_debug'))
adv_writer    = SummaryWriter(os.path.join(args.checkpoint_dir, 'adv_debug'))
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
log('==> Preparing data..')
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
    net      = ResNet34(num_classes=len(classes), activation=train_args['activation'])
    orig_net = ResNet34(num_classes=len(classes), activation=train_args['activation'])
elif train_args['net'] == 'resnet101':
    net      = ResNet101(num_classes=len(classes), activation=train_args['activation'])
    orig_net = ResNet101(num_classes=len(classes), activation=train_args['activation'])
else:
    raise AssertionError("network {} is unknown".format(train_args['net']))
net = net.to(device)
orig_net = orig_net.to(device)
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

# copying original net params to model_params:
reset_net()
reset_opt()
orig_net.load_state_dict(global_state['best_net'])
orig_params = torch.nn.utils.parameters_to_vector(orig_net.parameters())

def contrastive_loss(hidden, temperature=0.1):
    hidden1 = hidden[0:args.batch_size]
    hidden2 = hidden[args.batch_size:]
    hidden1 = nn.functional.normalize(hidden1)
    hidden2 = nn.functional.normalize(hidden2)
    cosine_sim = (1 - torch.matmul(hidden1, hidden2.T)).sum()
    return cosine_sim

def entropy_loss(logits):
    b = F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)
    b = -1.0 * b.sum()
    b = 1 / (eps + b)
    return b

def weight_diff_loss():
    global net, orig_net, orig_params
    diff = torch.tensor(0.0, requires_grad=True).to(device)
    torch.nn.utils.vector_to_parameters(orig_params, orig_net.parameters())
    for w1, w2 in zip(net.parameters(), orig_net.parameters()):
        diff = diff + torch.linalg.norm((w1 - w2).view(-1), ord=2)
    return diff

def get_debug(set, step):
    global loss_cont, loss_ent, loss_weight_diff
    if set == 'normal':
        # inputs
        x           = X_test

        # batch losses:
        loss_cont_d = loss_contrastive
        loss_ent_d  = loss_entropy
        loss_w_d    = loss_weight_difference

        # per image stats:
        cent_d      = cross_entropy
        ent_d       = entropy
        conf_d      = confidences

        # tta stats - simple:
        tta_cent_d  = tta_cross_entropy
        tta_ent_d   = tta_entropy
        tta_conf_d  = tta_confidences
        # tta stats - emb:
        tta_cent_emb_d  = tta_cross_entropy_emb
        tta_ent_emb_d   = tta_entropy_emb
        tta_conf_emb_d  = tta_confidences_emb
    else:
        # inputs
        x           = X_test_adv

        # batch losses:
        loss_cont_d = loss_contrastive_adv
        loss_ent_d  = loss_entropy_adv
        loss_w_d    = loss_weight_difference_adv

        # per image stats:
        cent_d      = cross_entropy_adv
        ent_d       = entropy_adv
        conf_d      = confidences_adv

        # tta stats - simple:
        tta_cent_d  = tta_cross_entropy_adv
        tta_ent_d   = tta_entropy_adv
        tta_conf_d  = tta_confidences_adv
        # tta stats - emb:
        tta_cent_emb_d  = tta_cross_entropy_emb_adv
        tta_ent_emb_d   = tta_entropy_emb_adv
        tta_conf_emb_d  = tta_confidences_emb_adv

    # collect last batch losses:
    loss_cont_d[img_ind, step] = loss_cont.item()
    loss_ent_d[img_ind, step] = loss_ent.item()
    loss_w_d[img_ind, step] = loss_weight_diff.item()

    # collect original image stats: cross-entropy, entropy, confidence:
    with torch.no_grad():
        x_tensor = torch.from_numpy(np.expand_dims(x[img_ind], 0)).to(device)
        y_tensor = torch.from_numpy(np.expand_dims(y_test[img_ind], 0)).to(device)
        out = net(x_tensor)
        cent_d[img_ind, step] = F.cross_entropy(out['logits'], y_tensor)
        ent_d[img_ind, step] = entropy_loss(out['logits'])
        conf_d[img_ind, step] = out['probs'].squeeze().max()

    # collect TTA images stats: cross-entropy, entropy, confidence
        emb_arr = -1 * torch.ones((args.tta_size, net.linear.weight.shape[1]),
                                  dtype=torch.float32, device=device, requires_grad=False)
        prob_arr = -1 * torch.ones((args.tta_size, len(classes)),
                                   dtype=torch.float32, device=device, requires_grad=False)
        tta_cnt = 0
        while tta_cnt < args.tta_size:
            (inputs, targets) = list(train_loader)[0]
            inputs, targets = inputs.to(device), targets.to(device)
            b = tta_cnt
            e = min(tta_cnt + len(inputs), args.tta_size)
            out = net(inputs)
            emb_arr[b:e] = out['embeddings'][0:(e-b)]
            prob_arr[b:e] = out['probs'][0:(e-b)]
            tta_cnt += e-b
            assert tta_cnt <= args.tta_size, 'not cool!'

        tta_probs = prob_arr.mean(axis=0)
        tta_probs = torch.unsqueeze(tta_probs, 0)
        tta_cent_d[img_ind, step] = F.cross_entropy(tta_probs, y_tensor)
        tta_ent_d[img_ind, step] = entropy_loss(tta_probs)
        tta_conf_d[img_ind, step] = tta_probs.squeeze().max()

        tta_emb_probs = get_probs_from_emb_center(emb_arr)
        tta_emb_probs = torch.unsqueeze(tta_emb_probs, 0)
        tta_cent_emb_d[img_ind, step] = F.cross_entropy(tta_emb_probs, y_tensor)
        tta_ent_emb_d[img_ind, step] = entropy_loss(tta_emb_probs)
        tta_conf_emb_d[img_ind, step] = tta_emb_probs.squeeze().max()

classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=contrastive_loss,
                               optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(classes))

img_cnt = NUM_DEBUG_SAMPLES if NUM_DEBUG_SAMPLES is not None else test_size

robustness_preds            = -1 * np.ones(test_size, dtype=np.int32)
robustness_preds_adv        = -1 * np.ones(test_size, dtype=np.int32)

# multi TTAs
robustness_probs            = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)
robustness_probs_adv        = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)
robustness_probs_emb        = -1 * np.ones((test_size, len(classes)), dtype=np.float32)
robustness_probs_emb_adv    = -1 * np.ones((test_size, len(classes)), dtype=np.float32)

# debug stats
# losses
loss_contrastive            = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
loss_contrastive_adv        = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
loss_entropy                = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
loss_entropy_adv            = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
loss_weight_difference      = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
loss_weight_difference_adv  = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
# per image stats
cross_entropy               = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
cross_entropy_adv           = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
entropy                     = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
entropy_adv                 = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
confidences                 = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
confidences_adv             = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
# tta stats - simple
tta_cross_entropy           = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
tta_cross_entropy_adv       = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
tta_entropy                 = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
tta_entropy_adv             = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
tta_confidences             = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
tta_confidences_adv         = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
# tta stats - emb
tta_cross_entropy_emb       = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
tta_cross_entropy_emb_adv   = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
tta_entropy_emb             = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
tta_entropy_emb_adv         = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
tta_confidences_emb         = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)
tta_confidences_emb_adv     = -1 * np.ones((img_cnt, args.steps + 1), dtype=np.float32)

def train(set):
    """set='normal' or 'adv'"""
    global TRAIN_TIME_CNT, loss_cont, loss_ent, loss_weight_diff

    start_time = time.time()
    reset_net()
    reset_proj()
    reset_opt()
    net.eval()
    proj_head.eval()

    for step in range(args.steps):
        (inputs, targets) = list(train_loader)[0]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = net(inputs)
        embeddings, logits = out['embeddings'], out['logits']
        z = proj_head(embeddings)
        loss_cont = contrastive_loss(z)
        loss_ent = entropy_loss(logits)
        loss_weight_diff = weight_diff_loss()
        loss = loss_cont + args.lambda_ent * loss_ent + args.lambda_wdiff * loss_weight_diff
        get_debug(set, step=step)
        loss.backward()
        optimizer.step()

    # for debug, last step:
    (inputs, targets) = list(train_loader)[0]
    inputs, targets = inputs.to(device), targets.to(device)
    out = net(inputs)
    embeddings, logits = out['embeddings'], out['logits']
    z = proj_head(embeddings)
    loss_cont = contrastive_loss(z)
    loss_ent = entropy_loss(logits)
    loss_weight_diff = weight_diff_loss()
    get_debug(set, step=args.steps)

    TRAIN_TIME_CNT += time.time() - start_time

def test(set):
    """set='normal' or 'adv'"""
    global TEST_TIME_CNT
    if set == 'normal':
        x = X_test
        rob_preds     = robustness_preds
        rob_probs     = robustness_probs
        rob_probs_emb = robustness_probs_emb
    else:
        x = X_test_adv
        rob_preds     = robustness_preds_adv
        rob_probs     = robustness_probs_adv
        rob_probs_emb = robustness_probs_emb_adv

    start_time = time.time()
    rob_preds[img_ind] = classifier.predict(np.expand_dims(x[img_ind], 0)).squeeze().argmax()
    with torch.no_grad():
        emb_arr = -1 * torch.ones((args.tta_size, net.linear.weight.shape[1]),
                                  dtype=torch.float32, device=device, requires_grad=False)
        tta_cnt = 0
        while tta_cnt < args.tta_size:
            (inputs, targets) = list(train_loader)[0]
            inputs, targets = inputs.to(device), targets.to(device)
            b = tta_cnt
            e = min(tta_cnt + len(inputs), args.tta_size)
            out = net(inputs)
            emb_arr[b:e] = out['embeddings'][0:(e-b)]
            rob_probs[img_ind, b:e] = out['probs'][0:(e-b)].detach().cpu().numpy()
            tta_cnt += e-b
            assert tta_cnt <= args.tta_size, 'not cool!'

        rob_probs_emb[img_ind] = get_probs_from_emb_center(emb_arr).detach().cpu().numpy()
    TEST_TIME_CNT += time.time() - start_time


for img_ind in tqdm(range(img_cnt)):
    # normal
    train_loader = get_single_img_dataloader(args.dataset, X_test, y_test, 2 * args.batch_size,
                                             pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    train('normal')
    test('normal')

    # adv
    train_loader = get_single_img_dataloader(args.dataset, X_test_adv, y_test, 2 * args.batch_size,
                                             pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    train('adv')
    test('adv')


if NUM_DEBUG_SAMPLES is not None:
    average_train_time = TRAIN_TIME_CNT / (2 * NUM_DEBUG_SAMPLES)
    average_test_time = TEST_TIME_CNT / (2 * NUM_DEBUG_SAMPLES)
else:
    average_train_time = TRAIN_TIME_CNT / (2 * test_size)
    average_test_time = TEST_TIME_CNT / (2 * test_size)
log('average train/test time per sample: {}/{} secs'.format(average_train_time, average_test_time))

if args.dump:
    log('dumping results...')
    # dumping args to txt file
    with open(os.path.join(DUMP_DIR, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    np.save(os.path.join(DUMP_DIR, 'robustness_preds.npy'), robustness_preds)
    np.save(os.path.join(DUMP_DIR, 'robustness_preds_adv.npy'), robustness_preds_adv)
    np.save(os.path.join(DUMP_DIR, 'robustness_probs.npy'), robustness_probs)
    np.save(os.path.join(DUMP_DIR, 'robustness_probs_adv.npy'), robustness_probs_adv)
    np.save(os.path.join(DUMP_DIR, 'robustness_probs_emb.npy'), robustness_probs_emb)
    np.save(os.path.join(DUMP_DIR, 'robustness_probs_emb_adv.npy'), robustness_probs_emb_adv)

    # debug
    np.save(os.path.join(DUMP_DIR, 'loss_contrastive.npy'), loss_contrastive)
    np.save(os.path.join(DUMP_DIR, 'loss_contrastive_adv.npy'), loss_contrastive_adv)
    np.save(os.path.join(DUMP_DIR, 'loss_entropy.npy'), loss_entropy)
    np.save(os.path.join(DUMP_DIR, 'loss_entropy_adv.npy'), loss_entropy_adv)
    np.save(os.path.join(DUMP_DIR, 'loss_weight_difference.npy'), loss_weight_difference)
    np.save(os.path.join(DUMP_DIR, 'loss_weight_difference_adv.npy'), loss_weight_difference_adv)

    np.save(os.path.join(DUMP_DIR, 'cross_entropy.npy'), cross_entropy)
    np.save(os.path.join(DUMP_DIR, 'cross_entropy_adv.npy'), cross_entropy_adv)
    np.save(os.path.join(DUMP_DIR, 'entropy.npy'), entropy)
    np.save(os.path.join(DUMP_DIR, 'entropy_adv.npy'), entropy_adv)
    np.save(os.path.join(DUMP_DIR, 'confidences.npy'), confidences)
    np.save(os.path.join(DUMP_DIR, 'confidences_adv.npy'), confidences_adv)

    np.save(os.path.join(DUMP_DIR, 'tta_cross_entropy.npy'), tta_cross_entropy)
    np.save(os.path.join(DUMP_DIR, 'tta_cross_entropy_adv.npy'), tta_cross_entropy_adv)
    np.save(os.path.join(DUMP_DIR, 'tta_entropy.npy'), tta_entropy)
    np.save(os.path.join(DUMP_DIR, 'tta_entropy_adv.npy'), tta_entropy_adv)
    np.save(os.path.join(DUMP_DIR, 'tta_confidences.npy'), tta_confidences)
    np.save(os.path.join(DUMP_DIR, 'tta_confidences_adv.npy'), tta_confidences_adv)

    np.save(os.path.join(DUMP_DIR, 'tta_cross_entropy_emb.npy'), tta_cross_entropy_emb)
    np.save(os.path.join(DUMP_DIR, 'tta_cross_entropy_emb_adv.npy'), tta_cross_entropy_emb_adv)
    np.save(os.path.join(DUMP_DIR, 'tta_entropy_emb.npy'), tta_entropy_emb)
    np.save(os.path.join(DUMP_DIR, 'tta_entropy_emb_adv.npy'), tta_entropy_emb_adv)
    np.save(os.path.join(DUMP_DIR, 'tta_confidences_emb.npy'), tta_confidences_emb)
    np.save(os.path.join(DUMP_DIR, 'tta_confidences_emb_adv.npy'), tta_confidences_emb_adv)

log('Calculating robustness metrics...')
calc_robust_metrics(robustness_preds, robustness_preds_adv)
calc_robust_metrics_from_probs_majority_vote(robustness_probs, robustness_probs_adv)
calc_robust_metrics_from_probs_summation(robustness_probs, robustness_probs_adv)
log('Calculating robustness metrics via embedding center...')
calc_robust_metrics(robustness_probs_emb.argmax(axis=1), robustness_probs_emb_adv.argmax(axis=1))

log('done')

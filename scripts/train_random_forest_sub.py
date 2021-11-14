"""
Train random forest substitute MLP using the normal images' probabilities (output of the random forest)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import scipy
import numpy as np
import json
import copy
import os
import argparse
import time
import pickle
import logging
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_loader_with_specific_inds, get_normalized_tensor
from active_learning_project.datasets.tta_utils import get_tta_transforms, get_tta_logits
from active_learning_project.datasets.utils import get_dataset_inds, get_ensemble_dir, get_dump_dir
from active_learning_project.datasets.tta_logits_dataset import TTALogitsDataset
from active_learning_project.utils import boolean_string, pytorch_evaluate, set_logger, get_ensemble_paths, \
    majority_vote, convert_tensor_to_image, print_Linf_dists, calc_attack_rate, get_image_shape
from active_learning_project.models.utils import get_strides, get_conv1_params, get_model
from active_learning_project.classifiers.pytorch_classifier_specific import PyTorchClassifierSpecific
from active_learning_project.models.mlp import MLP

parser = argparse.ArgumentParser(description='Evaluating robustness score')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name of teacher')
parser.add_argument('--random_forest_dir', default='random_forest', type=str, help='The dir which holds the RF params')
parser.add_argument('--sub_dir', default='sub_model', type=str, help='The dir which holds the substitute model')

parser.add_argument('--alpha', default=1, type=float, help='distillation ratio')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--mom', default=0.9, type=float, help='weight momentum of SGD optimizer')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')
parser.add_argument('--factor', default=0.75, type=float, help='LR schedule factor')
parser.add_argument('--patience', default=2, type=int, help='LR schedule patience')
parser.add_argument('--cooldown', default=0, type=int, help='LR cooldown')
parser.add_argument('--val_size', default=0.04, type=float, help='Fraction of validation size')
parser.add_argument('--num_workers', default=4, type=int, help='Data loading threads for tta loader or random forest')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
RF_DIR = os.path.join(args.checkpoint_dir, args.random_forest_dir)
SUB_DIR = os.path.join(RF_DIR, args.sub_dir)
SUB_PATH = os.path.join(SUB_DIR, 'ckpt.pth')
os.makedirs(SUB_DIR, exist_ok=True)
log_file = os.path.join(SUB_DIR, 'log.log')
set_logger(log_file)
logger = logging.getLogger()
rand_gen = np.random.RandomState(seed=12345)

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
batch_size = args.batch_size
dataset = train_args['dataset']
image_shape = get_image_shape(dataset)

# load logits of normal images:
logits = np.load(os.path.join(args.checkpoint_dir, 'normal', 'tta', 'tta_logits.npy'))
with open(os.path.join(args.checkpoint_dir, 'normal', 'tta', 'eval_args.txt'), 'r') as f:
    tta_args = json.load(f)
tta_args.update({'num_workers': args.num_workers})

# get RF predictions (probs):
rf_model_path = os.path.join(RF_DIR, 'random_forest_classifier.pkl')
with open(rf_model_path, "rb") as f:
    rf_model = pickle.load(f)
rf_model.n_jobs = None
rf_model.verbose = 0
rf_probs = rf_model.predict_proba(logits.reshape(logits.shape[0], -1))

# separating to train and val
def get_gt(dataset):
    tmp_loader = get_test_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False)
    X = get_normalized_tensor(tmp_loader, image_shape, batch_size)
    y_gt = np.asarray(tmp_loader.dataset.targets)
    classes = tmp_loader.dataset.classes
    return X, y_gt, classes

X, y_gt, classes = get_gt(dataset)
train_val_inds, test_inds = get_dataset_inds(dataset)
val_size = int(np.floor(args.val_size * len(train_val_inds)))
input_shape = (logits.shape[1], len(classes))

train_inds, val_inds = train_test_split(train_val_inds, test_size=val_size, random_state=rand_gen, shuffle=True,
                                        stratify=y_gt[train_val_inds])
train_inds.sort()
val_inds.sort()
np.save(os.path.join(SUB_DIR, 'sub_train_inds.npy'), train_inds)
np.save(os.path.join(SUB_DIR, 'sub_val_inds.npy'), val_inds)

# Set up train/val/test sets and loaders
logger.info('==> Preparing data..')
train_set = TTALogitsDataset(torch.from_numpy(logits[train_inds]), torch.from_numpy(rf_probs[train_inds]), torch.from_numpy(y_gt[train_inds]))
val_set   = TTALogitsDataset(torch.from_numpy(logits[val_inds]), torch.from_numpy(rf_probs[val_inds]), torch.from_numpy(y_gt[val_inds]))
test_set  = TTALogitsDataset(torch.from_numpy(logits[test_inds]), torch.from_numpy(rf_probs[test_inds]), torch.from_numpy(y_gt[test_inds]))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=device=='cuda')
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device=='cuda')
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=device=='cuda')

# set up network for data augmentation
logger.info('==> Building teacher model..')
conv1 = get_conv1_params(dataset)
strides = get_strides(dataset)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
if 'best_net' in global_state:
    global_state = global_state['best_net']
dnn = get_model(train_args['net'])(num_classes=len(classes), activation=train_args['activation'], conv1=conv1, strides=strides)
dnn = dnn.to(device)
dnn.load_state_dict(global_state)
dnn.eval()  # frozen

def generate_new_train_logits():
    global train_set, train_loader
    logger.info('Generating new train set TTA logits for epoch {}...'.format(epoch + 1))
    new_logits = get_tta_logits(dataset, dnn, X[train_inds], y_gt[train_inds], len(classes), tta_args)
    new_rf_probs = rf_model.predict_proba(new_logits.reshape(new_logits.shape[0], -1))
    train_set = TTALogitsDataset(torch.from_numpy(new_logits), torch.from_numpy(new_rf_probs), torch.from_numpy(y_gt[train_inds]))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device=='cuda')


train_writer = SummaryWriter(os.path.join(SUB_DIR, 'train'))
val_writer   = SummaryWriter(os.path.join(SUB_DIR, 'val'))
test_writer  = SummaryWriter(os.path.join(SUB_DIR, 'test'))

logger.info('==> Building model..')
net = MLP(len(classes))
net = net.to(device)
summary(net, input_shape)
if device == 'cuda':
    cudnn.benchmark = True

# losses
cross_entropy = nn.CrossEntropyLoss()

def kl(s_logits, t_probs):
    return F.kl_div(F.log_softmax(s_logits, dim=1), t_probs, reduction="batchmean")

def softXEnt(s_logits, t_probs):
    logprobs = F.log_softmax(s_logits, dim=1)
    return -(t_probs * logprobs).sum() / s_logits.shape[0]


WORST_METRIC = np.inf
metric_mode = 'min'
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd, nesterov=args.mom > 0)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode=metric_mode,
    factor=args.factor,
    patience=args.patience,
    verbose=True,
    cooldown=args.cooldown
)

# Training:
torch.autograd.set_detect_anomaly(True)

def save_global_state():
    global global_state
    global_state['best_net'] = copy.deepcopy(net).state_dict()
    global_state['best_metric'] = best_metric
    global_state['epoch'] = epoch
    global_state['global_step'] = global_step
    torch.save(global_state, SUB_PATH)

def save_current_state():
    torch.save(net.state_dict(), os.path.join(SUB_DIR, 'ckpt_epoch_{}.pth'.format(epoch)))

def flush():
    train_writer.flush()
    val_writer.flush()
    test_writer.flush()
    logger.handlers[0].flush()

def load_best_net():
    global net
    global_state = torch.load(SUB_PATH, map_location=torch.device(device))
    net.load_state_dict(global_state['best_net'])

def train():
    """Train and validate"""
    # Training
    global global_step
    global net

    net.train()
    train_loss = 0
    kl_loss = 0
    ce_loss = 0

    for batch_idx, (inputs, t_probs, targets) in enumerate(train_loader):  # train a single step
        inputs, t_probs, targets = inputs.to(device), t_probs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss_kl = kl(outputs['logits'].double(), t_probs)
        loss_ce = cross_entropy(outputs['logits'].double(), targets)
        loss = args.alpha * loss_kl + (1 - args.alpha) * loss_ce

        loss.backward()
        optimizer.step()
        kl_loss += loss_kl.item()
        ce_loss += loss_ce.item()
        train_loss += loss.item()

        if global_step % 10 == 0:  # sampling, once ever 10 train iterations
            train_writer.add_scalar('losses/kl', loss_kl.item(), global_step)
            train_writer.add_scalar('losses/ce', loss_ce.item(), global_step)
            train_writer.add_scalar('losses/loss', loss.item(), global_step)
            train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        global_step += 1

    N = batch_idx + 1
    train_loss = train_loss / N
    logger.info('Epoch #{} (TRAIN): kl_loss={}, ce_loss={}, loss={}'.format(epoch + 1, kl_loss, ce_loss, train_loss))

def validate():
    global global_state
    global best_metric

    net.eval()
    val_loss = 0
    kl_loss = 0
    ce_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, t_probs, targets) in enumerate(val_loader):
            inputs, t_probs, targets = inputs.to(device), t_probs.to(device), targets.to(device)
            outputs = net(inputs)
            loss_kl = kl(outputs['logits'].double(), t_probs)
            loss_ce = cross_entropy(outputs['logits'].double(), targets)
            loss = args.alpha * loss_kl + (1 - args.alpha) * loss_ce

            kl_loss += loss_kl.item()
            ce_loss += loss_ce.item()
            val_loss += loss.item()

    N = batch_idx + 1
    kl_loss = kl_loss / N
    ce_loss = ce_loss / N
    val_loss = val_loss / N

    val_writer.add_scalar('losses/kl', kl_loss, global_step)
    val_writer.add_scalar('losses/ce', ce_loss, global_step)
    val_writer.add_scalar('losses/loss', val_loss, global_step)
    metric = kl_loss

    if metric < best_metric:
        best_metric = metric
        logger.info('Found new best model. Saving...')
        save_global_state()

    logger.info('Epoch #{} (VAL): kl_loss={}, ce_loss={}, loss={},\tbest_metric({})={}'
                .format(epoch + 1, kl_loss, ce_loss, val_loss, 'KL div', best_metric))

    # updating learning rate if we see no improvement
    lr_scheduler.step(metrics=metric)

def test():
    # test
    net.eval()
    test_loss = 0
    kl_loss = 0
    ce_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, t_probs, targets) in enumerate(test_loader):
            inputs, t_probs, targets = inputs.to(device), t_probs.to(device), targets.to(device)
            outputs = net(inputs)
            loss_kl = kl(outputs['logits'].double(), t_probs)
            loss_ce = cross_entropy(outputs['logits'].double(), targets)
            loss = args.alpha * loss_kl + (1 - args.alpha) * loss_ce

            kl_loss += loss_kl.item()
            ce_loss += loss_ce.item()
            test_loss += loss.item()

    N = batch_idx + 1
    kl_loss = kl_loss / N
    ce_loss = ce_loss / N
    test_loss = test_loss / N

    test_writer.add_scalar('losses/kl', kl_loss, global_step)
    test_writer.add_scalar('losses/ce', ce_loss, global_step)
    test_writer.add_scalar('losses/loss', test_loss, global_step)
    logger.info('Epoch #{} (TEST): kl_loss={}, ce_loss={}, loss={}'.format(epoch + 1, kl_loss, ce_loss, test_loss))


best_metric    = WORST_METRIC
epoch          = 0
global_step    = 0
global_state = {}

logger.info('Testing epoch #{}'.format(epoch + 1))
test()

logger.info('start training from epoch #{} for {} epochs'.format(epoch + 1, args.epochs))
for epoch in tqdm(range(epoch, epoch + args.epochs)):
    # if epoch % 10 == 0 and epoch > 0:
    #     generate_new_train_logits()
    train()
    validate()
    test()
    flush()

# getting best metric, loading best net
load_best_net()
test()
flush()

exit(0)

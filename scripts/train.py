'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
from typing import Tuple, Any, Dict

import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import time
import sys
import logging
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")
sys.path.insert(0, "./TRADES")

from active_learning_project.models.resnet import ResNet34, ResNet50, ResNet101
from active_learning_project.models.wide_resnet_28_10 import WideResNet28_10
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader, \
    get_all_data_loader
from active_learning_project.utils import remove_substr_from_keys, boolean_string, save_features, pytorch_evaluate, \
    convert_tensor_to_image, get_image_shape, set_logger
from active_learning_project.models.utils import get_strides, get_conv1_params, get_model
from TRADES.trades import trades_loss

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='tiny_imagenet', type=str, help='dataset: cifar10, cifar100, svhn, tiny_imagenet')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--mom', default=0.9, type=float, help='weight momentum of SGD optimizer')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--net', default='resnet34', type=str, help='network architecture')
parser.add_argument('--activation', default='relu', type=str, help='network activation: relu or softplus')
parser.add_argument('--checkpoint_dir', default='/Users/giladcohen/data/gilad/logs/adv_robustness/debug', type=str, help='checkpoint dir')
parser.add_argument('--epochs', default='300', type=int, help='number of epochs')
parser.add_argument('--record', default=False, type=boolean_string, help='record all layers in each epoch')
parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')  # was 5e-4 for batch_size=128
parser.add_argument('--factor', default=0.9, type=float, help='LR schedule factor')
parser.add_argument('--patience', default=3, type=int, help='LR schedule patience')
parser.add_argument('--cooldown', default=0, type=int, help='LR cooldown')
parser.add_argument('--val_size', default=0.05, type=float, help='Fraction of validation size')
parser.add_argument('--num_workers', default=0, type=int, help='Data loading threads')
parser.add_argument('--metric', default='accuracy', type=str, help='metric to optimize. accuracy or sparsity')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--adv_trades', default=False, type=boolean_string, help='Use adv robust training using TRADES')

# TRADES params
parser.add_argument('--epsilon', default=0.031, type=float, help='epsilon for TRADES loss')
parser.add_argument('--step_size', default=0.007, type=float, help='step size for TRADES loss')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
log_file = os.path.join(args.checkpoint_dir, 'log.log')
os.makedirs(args.checkpoint_dir, exist_ok=True)
if args.record:
    os.makedirs(os.path.join(args.checkpoint_dir, 'records', 'trainval'), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, 'records', 'test'), exist_ok=True)
batch_size = args.batch_size

set_logger(log_file)
logger = logging.getLogger()
if args.metric == 'accuracy':
    WORST_METRIC = 0.0
    metric_mode = 'max'
elif args.metric == 'loss':
    WORST_METRIC = np.inf
    metric_mode = 'min'
else:
    raise AssertionError('illegal argument metric={}'.format(args.metric))

rand_gen = np.random.RandomState(int(time.time()))
train_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'train'))
val_writer   = SummaryWriter(os.path.join(args.checkpoint_dir, 'val'))
test_writer  = SummaryWriter(os.path.join(args.checkpoint_dir, 'test'))

# Data
logger.info('==> Preparing data..')

trainloader, valloader, train_inds, val_inds = get_train_valid_loader(
    dataset=args.dataset,
    batch_size=batch_size,
    rand_gen=rand_gen,
    valid_size=args.val_size,
    num_workers=args.num_workers,
    pin_memory=device=='cuda'
)
testloader = get_test_loader(
    dataset=args.dataset,
    batch_size=batch_size,
    num_workers=args.num_workers,
    pin_memory=device=='cuda'
)

if args.record:
    trainvalloader = get_all_data_loader(
        dataset=args.dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=device == 'cuda'
    )

img_shape = get_image_shape(args.dataset)
classes = trainloader.dataset.classes
train_size = len(trainloader.dataset)
val_size   = len(valloader.dataset)
test_size  = len(testloader.dataset)

# Model
logger.info('==> Building model..')
conv1 = get_conv1_params(args.dataset)
strides = get_strides(args.dataset)
net = get_model(args.net)(num_classes=len(classes), activation=args.activation, conv1=conv1, strides=strides)
net = net.to(device)
summary(net, (img_shape[2], img_shape[0], img_shape[1]))

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
y_train    = np.asarray(trainloader.dataset.targets)
y_val      = np.asarray(valloader.dataset.targets)
y_test     = np.asarray(testloader.dataset.targets)
if args.record:
    y_trainval = np.asarray(trainvalloader.dataset.targets)

# dump to dir:
np.save(os.path.join(args.checkpoint_dir, 'y_train.npy'), y_train)
np.save(os.path.join(args.checkpoint_dir, 'y_val.npy'), y_val)
np.save(os.path.join(args.checkpoint_dir, 'y_test.npy'), y_test)
np.save(os.path.join(args.checkpoint_dir, 'train_inds.npy'), train_inds)
np.save(os.path.join(args.checkpoint_dir, 'val_inds.npy'), val_inds)
if args.record:
    np.save(os.path.join(args.checkpoint_dir, 'y_trainval.npy'), y_trainval)

def reset_optim():
    global optimizer
    global lr_scheduler
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd, nesterov=args.mom > 0)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=metric_mode,
        factor=args.factor,
        patience=args.patience,
        verbose=True,
        cooldown=args.cooldown
    )

def reset_net():
    global net
    global global_step
    net.load_state_dict(global_state['best_net'])
    global_step = global_state['global_step']

def output_loss_robust(inputs, targets, is_training=False) -> Tuple[Dict, torch.Tensor]:
    global net
    global optimizer
    return trades_loss(net, inputs, targets, optimizer, args.step_size, args.epsilon, is_training=is_training)

def output_loss_normal(inputs, targets, is_training=False) -> Tuple[Dict, torch.Tensor]:
    global net
    global optimizer
    outputs = net(inputs)
    return outputs, criterion(outputs['logits'], targets)

if args.adv_trades:
    loss_func = output_loss_robust
else:
    loss_func = output_loss_normal

def train():
    """Train and validate"""
    # Training
    global global_state
    global global_step
    global epoch

    net.train()
    train_loss = 0
    predicted = []
    labels = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):  # train a single step
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, loss_ce = loss_func(inputs, targets, is_training=True)
        loss = loss_ce
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = outputs['logits'].max(1)
        predicted.extend(preds.cpu().numpy())
        labels.extend(targets.cpu().numpy())
        num_corrected = preds.eq(targets).sum().item()
        acc = num_corrected / targets.size(0)

        if global_step % 10 == 0:  # sampling, once ever 10 train iterations
            train_writer.add_scalar('losses/loss',    loss.item(),    global_step)
            train_writer.add_scalar('losses/loss_ce', loss_ce.item(), global_step)
            train_writer.add_scalar('metrics/acc', 100.0 * acc, global_step)
            train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        global_step += 1

    N = batch_idx + 1
    train_loss = train_loss / N
    predicted = np.asarray(predicted)
    labels = np.asarray(labels)
    train_acc = 100.0 * np.mean(predicted == labels)
    logger.info('Epoch #{} (TRAIN): loss={}\tacc={:.2f}'.format(epoch + 1, train_loss, train_acc))

def validate():
    global global_step
    global global_state
    global best_metric
    global epoch

    net.eval()
    val_loss = 0
    val_loss_ce = 0
    predicted = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, loss_ce = loss_func(inputs, targets, is_training=False)
            loss = loss_ce

            val_loss    += loss.item()
            val_loss_ce += loss_ce.item()

            predicted.extend(outputs['logits'].max(1)[1].cpu().numpy())

    N = batch_idx + 1
    val_loss    = val_loss / N
    val_loss_ce = val_loss_ce / N
    predicted = np.asarray(predicted)
    val_acc = 100.0 * np.mean(predicted == y_val)

    val_writer.add_scalar('losses/loss',    val_loss,    global_step)
    val_writer.add_scalar('losses/loss_ce', val_loss_ce, global_step)
    val_writer.add_scalar('metrics/acc', val_acc, global_step)

    if args.metric == 'accuracy':
        metric = val_acc
    elif args.metric == 'loss':
        metric = val_loss
    else:
        raise AssertionError('Unknown metric for optimization {}'.format(args.metric))

    if (args.metric == 'accuracy' and metric > best_metric) or (args.metric == 'loss' and metric < best_metric):
        best_metric = metric
        logger.info('Found new best model. Saving...')
        global_state['best_net'] = net.state_dict()
        global_state['best_metric'] = best_metric
        global_state['epoch'] = epoch
        global_state['global_step'] = global_step

    if epoch > 0 and epoch % 100 == 0:
        save_current_state()

    logger.info('Epoch #{} (VAL): loss={}\tacc={:.2f}\tbest_metric({})={}'.format(epoch + 1, val_loss, val_acc, args.metric, best_metric))

    # updating learning rate if we see no improvement
    lr_scheduler.step(metrics=metric, epoch=epoch)

def test():
    global global_state
    global global_step
    global epoch

    with torch.no_grad():
        # test
        net.eval()
        test_loss = 0
        test_loss_ce = 0
        predicted = []

        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, loss_ce = loss_func(inputs, targets, is_training=False)
            loss = loss_ce

            test_loss    += loss.item()
            test_loss_ce += loss_ce.item()

            predicted.extend(outputs['logits'].max(1)[1].cpu().numpy())

    N = batch_idx + 1
    test_loss    = test_loss / N
    test_loss_ce = test_loss_ce / N
    predicted = np.asarray(predicted)
    test_acc = 100.0 * np.mean(predicted == y_test)

    test_writer.add_scalar('losses/loss',    test_loss,    global_step)
    test_writer.add_scalar('losses/loss_ce', test_loss_ce, global_step)
    test_writer.add_scalar('metrics/acc', test_acc, global_step)

    logger.info('Epoch #{} (TEST): loss={}\tacc={:.2f}'.format(epoch + 1, test_loss, test_acc))

def record(subset):
    global epoch

    if subset == 'trainval':
        loader = trainvalloader
    elif subset == 'test':
        loader = testloader
    else:
        raise AssertionError('illegal subset = {}'.format(subset))

    fetch_keys = ['embeddings', 'logits']

    with torch.no_grad():
        outputs = pytorch_evaluate(net, loader, fetch_keys)

    for i, key in enumerate(fetch_keys):
        np.save(os.path.join(args.checkpoint_dir, 'records', subset, 'rec_{}_{}.npy'.format(epoch, key)), outputs[i])

def save_global_state():
    global epoch
    global global_state

    global_state['train_inds'] = train_inds
    global_state['val_inds'] = val_inds
    torch.save(global_state, CHECKPOINT_PATH)

def save_current_state():
    global epoch
    torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, 'ckpt_epoch_{}.pth'.format(epoch)))

def flush():
    train_writer.flush()
    val_writer.flush()
    test_writer.flush()

if __name__ == "__main__":
    if args.resume:
        # Load checkpoint.
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isfile(CHECKPOINT_PATH), 'Error: no checkpoint file found!'
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))

        # check if trained for DataParallel:
        if 'module' in list(checkpoint['best_net'].keys())[0]:
            checkpoint['best_net'] = remove_substr_from_keys(checkpoint['best_net'], 'module.')

        net.load_state_dict(checkpoint['best_net'])
        best_metric    = checkpoint['best_metric']
        epoch          = checkpoint['epoch']
        global_step    = checkpoint['global_step']
        train_inds     = checkpoint['train_inds']
        val_inds       = checkpoint['val_inds']

        global_state = checkpoint
    else:
        # no old knowledge
        best_metric    = WORST_METRIC
        epoch          = 0
        global_step    = 0
        # train_inds     = train_inds
        # val_inds       = val_inds

        global_state = {}

    # dumping args to txt file
    with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    reset_optim()

    logger.info('Testing epoch #{}'.format(epoch + 1))
    test()

    logger.info('start training from epoch #{} for {} epochs'.format(epoch + 1, args.epochs))
    for epoch in tqdm(range(epoch, epoch + args.epochs)):
        if args.record:
            record('trainval')
            record('test')
        train()
        validate()
        if epoch % 10 == 0 and epoch > 0:
            test()
            save_global_state()
    save_global_state()
    save_current_state()
    test()
    reset_net()
    test()  # post test the final best model
    flush()

exit(0)
# debug
# x_img = convert_tensor_to_image(inputs.detach().cpu().numpy())
# for i in range(10):
#     plt.imshow(x_img[i])
#     plt.show()
# labels = targets.detach().cpu().numpy()
# for i in range(10):
#     print(classes[labels[i]])

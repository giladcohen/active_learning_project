'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import time

from active_learning_project.models.resnet_v2 import ResNet18
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader
from active_learning_project.datasets.selection_methods import select_random, update_inds, SelectionMethodFactory
from active_learning_project.utils import remove_substr_from_keys
from torchsummary import summary

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/active_learning/debug', type=str, help='checkpoint dir')
parser.add_argument('--epochs', default='300', type=int, help='number of epochs')
parser.add_argument('--wd', default=0.00039, type=float, help='weight decay')  # was 5e-4 for batch_size=128
parser.add_argument('--factor', default=0.9, type=float, help='LR schedule factor')
parser.add_argument('--patience', default=3, type=int, help='LR schedule patience')
parser.add_argument('--cooldown', default=1, type=int, help='LR cooldown')
parser.add_argument('--val_size', default=0.05, type=float, help='Fraction of validation size')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/data/dataset/cifar10'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

rand_gen = np.random.RandomState(int(time.time()))
train_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'train'))
val_writer   = SummaryWriter(os.path.join(args.checkpoint_dir, 'val'))
test_writer  = SummaryWriter(os.path.join(args.checkpoint_dir, 'test'))

# Data
print('==> Preparing data..')

trainloader, valloader = get_train_valid_loader(
    data_dir=DATA_ROOT,
    batch_size=100,
    augment=True,
    rand_gen=rand_gen,
    valid_size=args.val_size,
    num_workers=1,
    pin_memory=device=='cuda'
)
testloader = get_test_loader(
    data_dir=DATA_ROOT,
    batch_size=100,
    num_workers=1,
    pin_memory=device=='cuda'
)

classes = trainloader.dataset.classes
dataset_size = len(trainloader.dataset)

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
summary(net, (3, 32, 32))

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def reset_optim():
    global optimizer
    global lr_scheduler
    global best_acc
    best_acc = 0.0
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.factor,
        patience=args.patience,
        verbose=True,
        cooldown=args.cooldown
    )

def reset_net():
    global net
    net.load_state_dict(global_state['best_net'])

def train():
    """Train and validate"""
    # Training
    global best_acc
    global global_state
    global global_step
    global epoch

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)['logits']
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if global_step % 10 == 0:  # once ever 100 train iterations
            train_writer.add_scalar('loss', train_loss/(batch_idx + 1), global_step)
            train_writer.add_scalar('acc', (100.0 * correct)/total, global_step)
            train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        global_step += 1

    train_loss = train_loss/(batch_idx + 1)
    train_acc = (100.0 * correct) / total
    print('Epoch #{} (TRAIN): loss={}\tacc={} ({}/{})'.format(epoch + 1, train_loss, train_acc, correct, total))

    # validation
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)['logits']
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss/(batch_idx + 1)
    val_acc = (100.0 * correct) / total

    val_writer.add_scalar('loss', val_loss, global_step)
    val_writer.add_scalar('acc', val_acc, global_step)

    if val_acc > best_acc:
        best_acc = val_acc
        print('Found new best model. Saving...')
        global_state['best_net'] = net.state_dict()
        global_state['best_acc'] = best_acc
        global_state['epoch'] = epoch
        global_state['global_step'] = global_step

    print('Epoch #{} (VAL): loss={}\tacc={} ({}/{})\tbest_acc={}'.format(epoch + 1, val_loss, val_acc, correct, total, best_acc))

    # updating learning rate if we see no improvement
    lr_scheduler.step(metrics=val_acc, epoch=epoch)

def test():
    global global_state
    global global_step
    global epoch

    with torch.no_grad():
        # test
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)['logits']
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_loss = test_loss / (batch_idx + 1)
        test_acc = (100.0 * correct) / total

        test_writer.add_scalar('loss', test_loss, global_step)
        test_writer.add_scalar('acc', test_acc, global_step)
        print('Epoch #{} (TEST): loss={}\tacc={} ({}/{})'.format(epoch + 1, test_loss, test_acc, correct, total))

def save_global_state():
    global epoch
    global global_state

    global_state['train_inds'] = train_inds
    global_state['val_inds'] = val_inds
    torch.save(global_state, CHECKPOINT_PATH)

if __name__ == "__main__":
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(CHECKPOINT_PATH), 'Error: no checkpoint file found!'
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))

        # check if trained for DataParallel:
        if 'module' in list(checkpoint['best_net'].keys())[0]:
            checkpoint['best_net'] = remove_substr_from_keys(checkpoint['best_net'], 'module.')

        net.load_state_dict(checkpoint['best_net'])
        best_acc       = checkpoint['best_acc']
        epoch          = checkpoint['epoch']
        global_step    = checkpoint['global_step']
        train_inds     = checkpoint['train_inds']
        val_inds       = checkpoint['val_inds']

        global_state = checkpoint
    else:
        # no old knowledge
        best_acc       = 0.0
        epoch          = 0
        global_step    = 0
        train_inds     = []
        val_inds       = []

        global_state = {}

    # dumping args to txt file
    with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    train_inds = trainloader.sampler.indices
    val_inds = valloader.sampler.data_source
    reset_optim()
    print('Testing epoch #{}'.format(epoch + 1))
    test()

    print('start training from epoch #{} for {} epochs'.format(epoch + 1, args.epochs))
    for epoch in tqdm(range(epoch, epoch + args.epochs)):
        train()
        if epoch % 10 == 0:
            test()
            save_global_state()

    save_global_state()
    reset_net()
    test()  # post test the final best model
    
    ###


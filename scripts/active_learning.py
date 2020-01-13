'''Train CIFAR10 with PyTorch.'''
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import sys
import json
import os
import argparse
from tqdm import tqdm
import time

from active_learning_project.models import *
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_loader_with_specific_inds, get_all_data_loader
from active_learning_project.datasets.selection_methods import select_random, update_inds, SelectionMethodFactory
from active_learning_project.utils import remove_substr_from_keys

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str, help='checkpoint dir')
parser.add_argument('--epochs', default='1500', type=int, help='number of epochs')
parser.add_argument('--wd', default=0.00039, type=float, help='weight decay')  # was 5e-4 for batch_size=128
parser.add_argument('--factor', default=0.9, type=float, help='LR schedule factor')
parser.add_argument('--patience', default=2, type=int, help='LR schedule patience')
parser.add_argument('--cooldown', default=1, type=int, help='LR cooldown')
parser.add_argument('--selection_method', default='farthest', type=str, help='Active learning index selection method')
parser.add_argument('--distance_norm', default='L2', type=str, help='Distance norm. Can be [L1/L2/L_inf]')
parser.add_argument('--include_val_as_train', '-i', action='store_true', help='Treats validation as train for AL')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/data/dataset/cifar10'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
ACTIVE_IND_DIR  = os.path.join(args.checkpoint_dir, 'active_indices')
BEST_CHECKPOINTS_DIR = os.path.join(args.checkpoint_dir, 'best_checkpoints')
SELECTION_EPOCHS = [300, 600, 900, 1200]
SELECTION_SIZE = 1000

rand_gen = np.random.RandomState(int(time.time()))
train_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'train'))
val_writer   = SummaryWriter(os.path.join(args.checkpoint_dir, 'val'))
test_writer  = SummaryWriter(os.path.join(args.checkpoint_dir, 'test'))

# Data
print('==> Preparing data..')

all_data_loader = get_all_data_loader(
    data_dir=DATA_ROOT,
    batch_size=100,
    num_workers=1,
    pin_memory=device=='cuda'
)

testloader = get_test_loader(
    data_dir=DATA_ROOT,
    batch_size=100,
    num_workers=1,
    pin_memory=device=='cuda'
)

classes = all_data_loader.dataset.classes
dataset_size = len(all_data_loader.dataset)

# Model
print('==> Building model..')
net = ResNet34()
net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
select = SelectionMethodFactory().config(args.selection_method)

# constructing select_args:
selection_args = {
    'selection_size': SELECTION_SIZE
}
if args.selection_method in ['farthest']:
    assert args.distance_norm in ['L1', 'L2', 'L_inf']
    selection_args.update({
        'distance_norm': args.distance_norm,
        'include_val_as_train': args.include_val_as_train
    })

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
    checkpoint_file = os.path.join(BEST_CHECKPOINTS_DIR, 'best_net_epoch_{}'.format(epoch))
    torch.save(global_state['best_net'], checkpoint_file)
    net.load_state_dict(global_state['best_net'])

def update_trainval_loaders(train_inds: list, val_inds: list):
    trainloader = get_loader_with_specific_inds(
        data_dir=DATA_ROOT,
        batch_size=100,
        is_training=True,
        indices=train_inds,
        num_workers=1,
        pin_memory=device=='cuda')

    valloader = get_loader_with_specific_inds(
        data_dir=DATA_ROOT,
        batch_size=100,
        is_training=False,
        indices=val_inds,
        num_workers=1,
        pin_memory=device=='cuda')

    return trainloader, valloader

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
    # update global_state
    global_state['net']         = net.state_dict()
    global_state['train_acc']   = train_acc
    global_state['global_step'] = global_step
    global_state['epoch']       = epoch

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
    # update global_state
    global_state['val_acc'] = val_acc

    if val_acc > best_acc:
        print('Found new best model. Saving...')
        best_acc = val_acc
        global_state['best_acc'] = val_acc
        global_state['best_net'] = net.state_dict()

    print('Epoch #{} (VAL): loss={}\tacc={} ({}/{})\tbest_acc={}'.format(epoch + 1, val_loss, val_acc, correct, total, best_acc))
    torch.save(global_state, CHECKPOINT_PATH)
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
        # updating state
        global_state['test_acc'] = test_acc
        torch.save(global_state, CHECKPOINT_PATH)

def save_current_inds(unlabeled_inds, train_inds, val_inds):
    global epoch
    global global_state

    np.save(os.path.join(ACTIVE_IND_DIR, 'unlabeled_inds_epoch_{}'.format(epoch)), unlabeled_inds)
    np.save(os.path.join(ACTIVE_IND_DIR, 'train_inds_epoch_{}'.format(epoch)), train_inds)
    np.save(os.path.join(ACTIVE_IND_DIR, 'val_inds_epoch_{}'.format(epoch)), val_inds)
    global_state['unlabeled_inds'] = unlabeled_inds
    global_state['train_inds'] = train_inds
    global_state['val_inds'] = val_inds

    # updating state
    torch.save(global_state, CHECKPOINT_PATH)

def get_inds_dict():
    return {'train_inds': train_inds, 'val_inds': val_inds, 'unlabeled_inds': unlabeled_inds}


if __name__ == "__main__":
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(CHECKPOINT_PATH), 'Error: no checkpoint file found!'
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))

        # check if trained for DataParallel:
        if 'module' in list(checkpoint['net'].keys())[0]:
            checkpoint['net'] = remove_substr_from_keys(checkpoint['net'], 'module.')

        net.load_state_dict(checkpoint['net'])
        best_acc       = checkpoint['best_acc']
        epoch          = checkpoint['epoch']
        global_step    = checkpoint['global_step']
        unlabeled_inds = checkpoint['unlabeled_inds']
        train_inds     = checkpoint['train_inds']
        val_inds       = checkpoint['val_inds']

        global_state = checkpoint
    else:
        # no old knowledge
        best_acc       = 0.0
        epoch          = 0
        global_step    = 0
        unlabeled_inds = np.arange(dataset_size).tolist()
        train_inds     = []
        val_inds       = []

        global_state = {}

    os.makedirs(ACTIVE_IND_DIR, exist_ok=True)
    os.makedirs(BEST_CHECKPOINTS_DIR, exist_ok=True)

    # dumping args to txt file
    with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print('start testing the model from epoch #{}...'.format(epoch + 1))
    if global_step == 0:
        print('initializing {} new random indices'.format(SELECTION_SIZE))
        inds_dict = get_inds_dict()
        init_inds = select_random(None, None, inds_dict, selection_args)
        update_inds(train_inds, val_inds, init_inds)
        unlabeled_inds = [ind for ind in range(dataset_size) if ind not in (train_inds + val_inds)]
        save_current_inds(unlabeled_inds, train_inds, val_inds)

    # initializing train and val loaders + optmizer
    trainloader, valloader = update_trainval_loaders(train_inds, val_inds)
    reset_optim()

    print('Testing epoch #{}'.format(epoch + 1))
    test()

    print('start training from epoch #{} for {} epochs'.format(epoch + 1, args.epochs))
    for epoch in tqdm(range(epoch, epoch + args.epochs)):
        if epoch in SELECTION_EPOCHS:
            print('Reached epoch #{}. Selecting {} new indices using {} method'.format(epoch + 1, SELECTION_SIZE, args.selection_method))
            inds_dict = get_inds_dict()
            new_inds = select(net, all_data_loader, inds_dict, cfg=selection_args)  # dataset w/o augmentations
            update_inds(train_inds, val_inds, new_inds)
            unlabeled_inds = [ind for ind in range(dataset_size) if ind not in (train_inds + val_inds)]
            save_current_inds(unlabeled_inds, train_inds, val_inds)

            # initializing train and val loaders + optmizer + checkpoint (to best checkpoint)
            trainloader, valloader = update_trainval_loaders(train_inds, val_inds)
            reset_optim()
            reset_net()

        train()

        if epoch % 10 == 0:
            test()
    test()  # post test the final model without training


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

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")


from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader
from active_learning_project.utils import remove_substr_from_keys
from torchsummary import summary

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='svhn', type=str, help='dataset: cifar10, cifar100, svhn')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--mom', default=0.9, type=float, help='weight momentum of SGD optimizer')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--net', default='resnet34', type=str, help='network architecture')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/adv_robustness/svhn/debug', type=str, help='checkpoint dir')
parser.add_argument('--epochs', default='300', type=int, help='number of epochs')
parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')  # was 5e-4 for batch_size=128
parser.add_argument('--factor', default=0.9, type=float, help='LR schedule factor')
parser.add_argument('--patience', default=3, type=int, help='LR schedule patience')
parser.add_argument('--cooldown', default=0, type=int, help='LR cooldown')
parser.add_argument('--val_size', default=0.05, type=float, help='Fraction of validation size')
parser.add_argument('--n_workers', default=1, type=int, help='Data loading threads')
parser.add_argument('--metric', default='accuracy', type=str, help='metric to optimize. accuracy or sparsity')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
os.makedirs(args.checkpoint_dir, exist_ok=True)
batch_size = args.batch_size

rand_gen = np.random.RandomState(int(time.time()))
train_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'train'))
val_writer   = SummaryWriter(os.path.join(args.checkpoint_dir, 'val'))
test_writer  = SummaryWriter(os.path.join(args.checkpoint_dir, 'test'))

# Data
print('==> Preparing data..')

trainloader, valloader, train_inds, val_inds = get_train_valid_loader(
    dataset=args.dataset,
    batch_size=batch_size,
    rand_gen=rand_gen,
    valid_size=args.val_size,
    num_workers=args.n_workers,
    pin_memory=device=='cuda'
)
testloader = get_test_loader(
    dataset=args.dataset,
    batch_size=batch_size,
    num_workers=args.n_workers,
    pin_memory=device=='cuda'
)

classes = trainloader.dataset.classes
train_size = len(trainloader.dataset)
val_size   = len(valloader.dataset)
test_size  = len(testloader.dataset)

# Model
print('==> Building model..')
if args.net == 'resnet34':
    net = ResNet34(num_classes=len(classes))
elif args.net == 'resnet101':
    net = ResNet101(num_classes=len(classes))
else:
    raise AssertionError("network {} is unknown".format(args.net))

net = net.to(device)
summary(net, (3, 32, 32))

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
y_val = np.asarray(valloader.dataset.targets)
y_test = np.asarray(testloader.dataset.targets)

def reset_optim():
    global optimizer
    global lr_scheduler
    global best_metric
    best_metric = 0.0
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd, nesterov=args.mom > 0)
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
    global global_step
    net.load_state_dict(global_state['best_net'])
    global_step = global_state['global_step']

def train():
    """Train and validate"""
    # Training
    global best_metric
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
        outputs = net(inputs)
        loss_ce = criterion(outputs['logits'], targets)
        loss = loss_ce
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = outputs['logits'].max(1)
        predicted.extend(preds.cpu().numpy())
        labels.extend(targets.cpu().numpy())
        num_corrected = preds.eq(targets).sum().item()
        acc = num_corrected / targets.size(0)

        if global_step % 10 == 0:  # sampling, once ever 100 train iterations
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
    print('Epoch #{} (TRAIN): loss={}\tacc={:.2f}'
          .format(epoch + 1, train_loss, train_acc))

    # validation
    net.eval()
    val_loss = 0
    val_loss_ce = 0
    predicted = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss_ce = criterion(outputs['logits'], targets)
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
    else:
        raise AssertionError('Unknown metric for optimization {}'.format(args.metric))

    if metric > best_metric:
        best_metric = metric
        print('Found new best model. Saving...')
        global_state['best_net'] = net.state_dict()
        global_state['best_metric'] = best_metric
        global_state['epoch'] = epoch
        global_state['global_step'] = global_step

    print('Epoch #{} (VAL): loss={}\tacc={:.2f}\tbest_metric({})={:.4f}'
          .format(epoch + 1, val_loss, val_acc, args.metric, best_metric))

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
            outputs = net(inputs)
            loss_ce = criterion(outputs['logits'], targets)
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

    print('Epoch #{} (TEST): loss={}\tacc={:.2f}'
          .format(epoch + 1, test_loss, test_acc))

def save_global_state():
    global epoch
    global global_state

    global_state['train_inds'] = train_inds
    global_state['val_inds'] = val_inds
    torch.save(global_state, CHECKPOINT_PATH)

def flush():
    train_writer.flush()
    val_writer.flush()
    test_writer.flush()

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
        best_metric    = checkpoint['best_metric']
        epoch          = checkpoint['epoch']
        global_step    = checkpoint['global_step']
        train_inds     = checkpoint['train_inds']
        val_inds       = checkpoint['val_inds']

        global_state = checkpoint
    else:
        # no old knowledge
        best_metric    = 0.0
        epoch          = 0
        global_step    = 0
        # train_inds     = train_inds
        # val_inds       = val_inds

        global_state = {}

    # dumping args to txt file
    with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

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
    test()
    reset_net()
    test()  # post test the final best model
    flush()

# This module is adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import os
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import numpy as np
from torchsummary import summary
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")
sys.path.insert(0, "./FreeAdversarialTraining")
sys.path.insert(0, "./FreeAdversarialTraining/lib")
from utils import *

from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader

DATA_DIR = '/Users/giladcohen/data/dataset/cifar10/try1_270720'
RESUME = False
EVALUATE = False
PRETRAINED = False
OUTPUT_DIR = '/Users/giladcohen/logs/debug'
TRAIN_EPOCHES = 8
ADV_N_REPEATS = 4
FGSM_STEP = 4.0
MAX_COLOR_VALUE = 255.0
CROP_SIZE = 32
CLIP_EPS = 4.0
LR = 0.1
MOMENTUM = 0.9
WD = 0.0001  # TODO(change)
BATCH_SIZE = 100
NUM_WORKERS = 4  # TODO(change)
START_EPOCH = 0
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2471, 0.2435, 0.2616]
PRINT_FREQ = 10
ADV_PGD_ATTACK = [[10, 1/255], [50, 1/255]]  # 10 iters and 50 iters
FACTOR = 0.9
PATIENCE = 3
COOLDOWN = 0

VAL_SIZE = 0.05

rand_gen = np.random.RandomState(15101985)
train_writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'train'))
val_writer   = SummaryWriter(os.path.join(OUTPUT_DIR, 'val'))

# rand_gen = np.random.RandomState(int(time.time()))

# Parse config file and initiate logging
logger = initiate_logger(OUTPUT_DIR)
print = logger.info
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

# Scale and initialize the parameters
best_acc = 0.0
TRAIN_EPOCHES = int(math.ceil(TRAIN_EPOCHES / ADV_N_REPEATS))
FGSM_STEP /= MAX_COLOR_VALUE
CLIP_EPS /= MAX_COLOR_VALUE

# Create output folder
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("=> creating model '{}'".format('Resnet34'))
net = ResNet34(num_classes=10)
net = net.to(device)
summary(net, (3, 32, 32))

# Criterion:
criterion = nn.CrossEntropyLoss().cuda()

# Optimizer:
#TODO(add nesterov)
optimizer = torch.optim.SGD(net.parameters(), LR, momentum=MOMENTUM, weight_decay=WD)

# Resume if a valid checkpoint path is provided
#TODO(support resume)

# Initiate data loaders
train_loader, val_loader, train_inds, val_inds = get_train_valid_loader(
    dataset='cifar10',
    batch_size=BATCH_SIZE,
    rand_gen=rand_gen,
    valid_size=VAL_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=device=='cuda'
)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=FACTOR,
        patience=PATIENCE,
        verbose=True,
        cooldown=COOLDOWN
    )

# If in evaluate mode: perform validation on PGD attacks as well as clean samples
#TODO(add validation)

global global_noise_data
global_noise_data = torch.zeros([BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE]).cuda()
def train(train_loader, net, criterion, optimizer, epoch):
    global global_noise_data
    mean = torch.Tensor(np.array(MEAN)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, CROP_SIZE, CROP_SIZE).cuda()
    std = torch.Tensor(np.array(STD)[:, np.newaxis, np.newaxis])
    std = std.expand(3, CROP_SIZE, CROP_SIZE).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    # switch to train mode
    net.train()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)
        for j in range(ADV_N_REPEATS):
            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)
            output = net(in1)['logits']
            loss = criterion(output, target)

            accu = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            acc.update(accu[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, FGSM_STEP)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-CLIP_EPS, CLIP_EPS)

            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                print('Train Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f}\t'
                      'Data {data_time.val:.3f}\t'
                      'Loss {cls_loss.val:.4f}\t'
                      'Acc {acc.val:.3f}'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, acc=acc, cls_loss=losses))
                sys.stdout.flush()


def validate(val_loader, net, criterion, logger):
    # Mean/Std for normalization
    mean = torch.Tensor(np.array(MEAN)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, CROP_SIZE, CROP_SIZE).cuda()
    std = torch.Tensor(np.array(STD)[:, np.newaxis, np.newaxis])
    std = std.expand(3, CROP_SIZE, CROP_SIZE).cuda()

    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    # switch to evaluate mode
    net.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            input = input - mean
            input.div_(std)
            output = net(input)['logits']
            loss = criterion(output, target)

            # measure accuracy and record loss
            accu = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            acc.update(accu[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'acc {acc.val:.3f}\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))
                sys.stdout.flush()

    print('Final acc: {acc.avg:.3f}'.format(acc=acc))
    return acc.avg


def validate_pgd(val_loader, net, criterion, K, step, logger):
    # Mean/Std for normalization
    mean = torch.Tensor(np.array(MEAN)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, CROP_SIZE, CROP_SIZE).cuda()
    std = torch.Tensor(np.array(STD)[:, np.newaxis, np.newaxis])
    std = std.expand(3, CROP_SIZE, CROP_SIZE).cuda()
    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    eps = CLIP_EPS
    net.eval()
    end = time.time()
    logger.info(pad_str(' PGD eps: {}, K: {}, step: {} '.format(eps, K, step)))
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        orig_input = input.clone()
        randn = torch.FloatTensor(input.size()).uniform_(-eps, eps).cuda()
        input += randn
        input.clamp_(0, 1.0)
        for _ in range(K):
            invar = Variable(input, requires_grad=True)
            in1 = invar - mean
            in1.div_(std)
            output = net(in1)['logits']
            ascend_loss = criterion(output, target)
            ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
            pert = fgsm(ascend_grad, step)
            # Apply purturbation
            input += pert.data
            input = torch.max(orig_input - eps, input)
            input = torch.min(orig_input + eps, input)
            input.clamp_(0, 1.0)

        input.sub_(mean).div_(std)
        with torch.no_grad():
            # compute output
            output = net(input)['logits']
            loss = criterion(output, target)

            # measure accuracy and record loss
            accu = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            acc.update(accu[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                print('PGD Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'acc {acc.val:.3f}'.format(i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))
                sys.stdout.flush()

    print(' PGD Final acc {acc.avg:.3f}'.format(acc=acc))
    return acc.avg


for epoch in range(START_EPOCH, TRAIN_EPOCHES):
    # adjust_learning_rate(configs.TRAIN.lr, optimizer, epoch, configs.ADV.n_repeats)

    # train for one epoch
    train(train_loader, net, criterion, optimizer, epoch)

    # evaluate on validation set
    acc = validate(val_loader, net, criterion, logger)

    # remember best acc and save checkpoint
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    if is_best:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'Resnet34',
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, OUTPUT_DIR)
        lr_scheduler.step(metrics=best_acc, epoch=epoch)

# Automatically perform PGD Attacks at the end of training
logger.info(pad_str(' Performing PGD Attacks '))
for pgd_param in ADV_PGD_ATTACK:
    validate_pgd(val_loader, net, criterion, pgd_param[0], pgd_param[1], logger)

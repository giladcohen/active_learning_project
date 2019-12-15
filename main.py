'''Train CIFAR10 with PyTorch.'''
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np

import os
import argparse
from tqdm import tqdm

from active_learning_project.models import *
from active_learning_project.datasets.train_val_test_data_loaders import get_train_valid_loader, get_test_loader

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str, help='checkpoint dir')
parser.add_argument('--epochs', default='200', type=int, help='number of epochs')
parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
parser.add_argument('--factor', default=0.9, type=float, help='LR schedule factor')
parser.add_argument('--cooldown', default=2, type=int, help='LR cooldown')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/data/dataset/cifar10'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
ACTIVE_IND_DIR  = os.path.join(args.checkpoint_dir, 'active_indices')

rand_gen = np.random.RandomState(12345)

# Data
print('==> Preparing data..')
trainloader, valloader = get_train_valid_loader(
    data_dir=DATA_ROOT,
    batch_size=128,
    augment=True,
    rand_gen=rand_gen,
    valid_size=0.1,
    num_workers=1,
    pin_memory=device=='cuda'
)
testloader = get_test_loader(
    data_dir=DATA_ROOT,
    batch_size=100,
    shuffle=False,
    num_workers=1,
    pin_memory=device=='cuda'
)

classes = trainloader.dataset.classes

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = ResNet34()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=args.factor,
    patience=3,
    verbose=True,
    cooldown=2
)

def train():
    """Train and validate"""
    # Training
    global best_acc
    global global_state
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = train_loss/(batch_idx + 1)
    train_acc = (100.0 * correct) / total
    print('Epoch #{} (TRAIN): loss={}\tacc={} ({}/{})'.format(epoch, train_loss, train_acc, correct, total))

    # validation
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss/(batch_idx + 1)
    val_acc = (100.0 * correct) / total
    print('Epoch #{} (VAL): loss={}\tacc={} ({}/{})\tbest_acc={}'.format(epoch, val_loss, val_acc, correct, total, best_acc))

    if val_acc > best_acc:
        print('Found new best model. Saving...')
        state = {
            'net': net.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'epoch': epoch,
        }
        global_state.update(state)
        torch.save(state, CHECKPOINT_PATH)
        best_acc = val_acc

    # updating learning rate if we see no improvement
    lr_scheduler.step(metrics=val_acc, epoch=epoch)

def test():
    global global_state
    with torch.no_grad():
        # test
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_loss = test_loss / (batch_idx + 1)
        test_acc = (100.0 * correct) / total
        state = {
            'test_acc': test_acc,
        }
        global_state.update(state)
        torch.save(global_state, CHECKPOINT_PATH)

        print('Epoch #{} (TEST): loss={}\tacc={} ({}/{})'.format(epoch, test_loss, test_acc, correct, total))


if __name__ == 'main':

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(CHECKPOINT_PATH), 'Error: no checkpoint file found!'
        checkpoint = torch.load(CHECKPOINT_PATH)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['val_acc']
        start_epoch = checkpoint['epoch']
        global_state = checkpoint
    else:
        # no old knowledge
        best_acc = 0.0
        start_epoch = 0
        global_state = {}

    os.makedirs(ACTIVE_IND_DIR, exist_ok=True)  # checkpoint file found, or starting new training folder

    print('start testing the model for the first time...')
    epoch = start_epoch
    test()  # pretest the random model without training

    print('start training from epoch #{} for {} epochs'.format(start_epoch + 1, args.epochs))
    for epoch in tqdm(range(start_epoch+1, start_epoch + args.epochs)):
        # saving indices of train and val sets
        train_inds_np = np.asarray(trainloader.sampler.indices)
        tval_inds_np  = np.asarray(valloader.sampler.indices)
        np.save(os.path.join(ACTIVE_IND_DIR, 'train_inds_epoch_{}'.format(epoch)))
        np.save(os.path.join(ACTIVE_IND_DIR, 'val_inds_epoch_{}'.format(epoch)))
        train()
        if epoch % 10 == 0:
            test()
    test()  # post test the final model without training


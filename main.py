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
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
DATA_ROOT = '/data/dataset/cifar10'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
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

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpoint_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=args.factor,
    patience=3,
    verbose=True,
    cooldown=2
)

# Training
best_acc = 0.0
for epoch in tqdm(range(args.epochs)):
    print('\nEpoch: %d' % epoch)
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
        acc = (100.0 * correct) / total

    lr_scheduler.step(metrics=acc, epoch=epoch)
    loss_mean = train_loss/(batch_idx + 1)
    print('Epoch: {}: loss={}\taccuracy={} ({}/{})'.format(epoch, loss_mean, acc, correct, total))

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(state, CHECKPOINT_PATH)
        best_acc = acc


# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#
#             # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#             #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
#
#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.pth')
#         best_acc = acc

# for epoch in range(start_epoch, start_epoch+200):
#     train(epoch)
#     test(epoch)

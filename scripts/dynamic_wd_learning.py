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


from active_learning_project.models.resnet_v2 import ResNet18
from active_learning_project.models.jakubovitznet import JakubovitzNet
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader
from active_learning_project.utils import remove_substr_from_keys
from torchsummary import summary
from active_learning_project.utils import boolean_string
from adversarial_robustness_toolbox.art.classifiers import PyTorchClassifier
from cleverhans.utils import random_targets, to_categorical
from adversarial_robustness_toolbox.art.attacks import FastGradientMethod

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--mom', default=0.9, type=float, help='weight momentum of SGD optimizer')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--net', default='resnet', type=str, help='network architecture')
parser.add_argument('--checkpoint_dir', default='/disk4/dynamic_wd/debug104', type=str, help='checkpoint dir')
parser.add_argument('--epochs', default='250', type=int, help='number of epochs')
parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')  # was 5e-4 for batch_size=128
parser.add_argument('--ad', default=0.01, type=float, help='activation decay')
parser.add_argument('--use_bn', default=True, type=boolean_string, help='whether or not to use batch norm')
parser.add_argument('--factor', default=0.9, type=float, help='LR schedule factor')
parser.add_argument('--patience', default=3, type=int, help='LR schedule patience')
parser.add_argument('--cooldown', default=0, type=int, help='LR cooldown')
parser.add_argument('--val_size', default=0.05, type=float, help='Fraction of validation size')
parser.add_argument('--n_workers', default=1, type=int, help='Data loading threads')
parser.add_argument('--metric', default='sparsity', type=str, help='metric to optimize. accuracy or sparsity')
parser.add_argument('--debug', '-d', action='store_true', help='debug logs and dumps')
parser.add_argument('--attack', default='fgsm', type=str, help='checkpoint dir')
parser.add_argument('--targeted', default=True, type=boolean_string, help='use targeted attack')


parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/data/dataset/cifar10'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack)
if args.targeted:
    ATTACK_DIR = ATTACK_DIR + '_targeted'
os.makedirs(ATTACK_DIR, exist_ok=True)

BATCH_SIZE = 100

rand_gen = np.random.RandomState(int(time.time()))
train_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'train'))
val_writer   = SummaryWriter(os.path.join(args.checkpoint_dir, 'val'))
test_writer  = SummaryWriter(os.path.join(args.checkpoint_dir, 'test'))

# Data
print('==> Preparing data..')

trainloader, valloader, train_inds, val_inds = get_train_valid_loader(
    data_dir=DATA_ROOT,
    batch_size=BATCH_SIZE,
    augment=True,
    rand_gen=rand_gen,
    valid_size=args.val_size,
    num_workers=args.n_workers,
    pin_memory=device=='cuda'
)
testloader = get_test_loader(
    data_dir=DATA_ROOT,
    batch_size=BATCH_SIZE,
    num_workers=args.n_workers,
    pin_memory=device=='cuda'
)

classes = trainloader.dataset.classes
train_size = len(trainloader.dataset)
val_size   = len(valloader.dataset)
test_size  = len(testloader.dataset)

# Model
print('==> Building model..')
if args.net == 'jaku':
    net = JakubovitzNet(num_classes=len(classes))
elif args.net == 'resnet':
    net = ResNet18(num_classes=len(classes), use_bn=args.use_bn)
else:
    raise AssertionError("network {} is unknown".format(args.net))

net = net.to(device)
summary(net, (3, 32, 32))

def inverse_map(x: dict) -> dict:
    """
    :param x: dictionary listing for each key (relu activation) the relevant weight which are its input
    :return: inverse mapping, showing for each weight param what "num_act" key to consider
    """
    inv_map = {}
    for k, v in x.items():
        for w in v:
            inv_map[w] = k
    return inv_map


weight_reg_map = inverse_map(net.weight_reg_dict)

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
base_wd = torch.tensor(args.wd)
base_ad = torch.tensor(args.ad)
NUM_LAYERS = torch.tensor(17)  # TODO(gilad): Set for each layer

def reset_optim():
    global optimizer
    global lr_scheduler
    global best_metric
    best_metric = 0.0
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=0.0, nesterov=args.mom > 0)
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

def add_to_tensor(t: torch.tensor, x: torch.tensor) -> torch.tensor:
    """
    :param t: Tensor to add to
    :param x: addition
    :return: t + x. If t is None, returns x.
    """
    if t is None:
        t = x
    else:
        t = t + x

    return t

def activation_decay(outputs):
    """
    :param outputs: net output dictionary
    :return: scalar differentiable tensor, activation decay multiplied by ratio of activations, dict of losses for debug
    """
    # if args.ad == 0.0:
    #     return torch.tensor(0.0), {}

    ad_dict = {}
    # l_reg = torch.tensor(0.0, requires_grad=True)
    l_reg = None  # setting like this because it will automatically determine if we require grads or not

    for key, val in outputs.items():
        if 'L1_act' in key:
            ad_dict[key] = val.item()
            l_reg = add_to_tensor(l_reg, val)
    l_reg = (l_reg / NUM_LAYERS) * base_ad
    return l_reg, ad_dict

def activation_sparsity(outputs):
    """
    :param outputs: net output dictionary
    :return: dict. sparse metric for each layer.
    """
    sp_dict = {}
    for key, val in outputs.items():
        if 'num_act' in key:
            sp_dict[key] = 1.0 - val.item()
    return sp_dict

def calc_sparsity(s_dict):
    """
    :param s_dict: sparsity dict for all the layers
    :return: sparsity metric score
    """
    sparsity_tot = 0.0
    for key, val in s_dict.items():
        sparsity_tot += val
    score = sparsity_tot / NUM_LAYERS.item()
    return score

def weight_decay(net):
    """
    :param net: network
    :return: scalar differentiable tensor, weight decay multiplied by ratio of activations.
    """
    # l_reg = torch.tensor(0.0, requires_grad=True)
    l_reg = None  # setting like this because it will automatically determine if we require grads or not
    for name, params in net.named_parameters():
        l_reg = add_to_tensor(l_reg, 0.5 * (params**2).sum())
    l_reg = l_reg * base_wd
    return l_reg

def collect_debug(writer, ad_dict, sp_dict, N=1, grads=False):
    for key in ad_dict.keys():
        val = ad_dict[key] / N
        writer.add_scalar('losses/loss_ad_{}'.format(key[6:]), val * args.ad, global_step)
    for key in sp_dict.keys():
        val = sp_dict[key] / N
        writer.add_scalar('metrics/sparsity_{}'.format(key[7:]), val, global_step)
    if grads:  # should be used only during training
        assert N == 1
        ad_grads_dict = {}
        for name, params in net.named_parameters():
            if name in weight_reg_map.keys():
                key = weight_reg_map[name]
                ad_grads_dict[key] = ad_grads_dict.get(key, 0) + params.grad.abs().mean().item()
        for key, val in ad_grads_dict.items():
            writer.add_scalar('loss_grads/grad_ad_{}'.format(key[3:]), val, global_step)

def calc_robustness(X, predicted, targets, adv_targets=None):
    """
    Calculating robustness of network to attack
    :param X: dataset
    :param predicted: predicted labels
    :param adv_targets: adversarial targets
    :return: float, robustness
    """
    net.eval()
    n = X.shape[0]
    X_adv = attack.generate(x=X, y=adv_targets)
    adv_logits = classifier.predict(X_adv, batch_size=BATCH_SIZE)
    adv_preds = np.argmax(adv_logits, axis=1)
    adv_accuracy = 100.0 * np.sum(adv_preds == targets) / n

    # calculate attack rate
    info = {}
    for i in range(n):
        info[i] = {}
        info[i]['net_succ']    = predicted[i] == targets[i]
        info[i]['attack_succ'] = predicted[i] != adv_preds[i]

    net_succ_indices             = [ind for ind in info if info[ind]['net_succ']]
    net_succ_attack_succ_indices = [ind for ind in info if info[ind]['net_succ'] and info[ind]['attack_succ']]
    attack_rate = len(net_succ_attack_succ_indices) / len(net_succ_indices)
    robustness = 1.0 - attack_rate

    return adv_accuracy, robustness

def train():
    """Train and validate"""
    # Training
    global best_metric
    global global_state
    global global_step
    global epoch

    net.train()
    train_loss = 0
    train_sparsity = 0
    acc_forward_time = 0.0
    acc_backward_time = 0.0
    predicted = []
    labels = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):  # train a single step
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        start = time.time()
        outputs = net(inputs)
        loss_ce = criterion(outputs['logits'], targets)
        loss_wd = weight_decay(net)
        loss_ad, ad_dict = activation_decay(outputs)
        sp_dict = activation_sparsity(outputs)
        loss = loss_ce + loss_wd + loss_ad
        end = time.time()
        acc_forward_time += (end - start)*(batch_idx > 0)
        start = time.time()
        loss.backward(retain_graph=args.debug)
        end = time.time()
        acc_backward_time += (end - start)*(batch_idx > 0)
        optimizer.step()

        train_loss += loss.item()

        _, preds = outputs['logits'].max(1)
        predicted.extend(preds.cpu().numpy())
        labels.extend(targets.cpu().numpy())
        num_corrected = preds.eq(targets).sum().item()
        acc = num_corrected / targets.size(0)

        sparsity = calc_sparsity(sp_dict)  # for current iter calculation
        train_sparsity += sparsity

        if global_step % 10 == 0:  # sampling, once ever 100 train iterations
            train_writer.add_scalar('losses/loss',    loss.item(),    global_step)
            train_writer.add_scalar('losses/loss_ce', loss_ce.item(), global_step)
            train_writer.add_scalar('losses/loss_wd', loss_wd.item(), global_step)
            train_writer.add_scalar('losses/loss_ad', loss_ad.item(), global_step)

            train_writer.add_scalar('metrics/acc', 100.0 * acc, global_step)
            train_writer.add_scalar('metrics/sparsity', sparsity, global_step)
            train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            if args.debug:
                collect_debug(train_writer, ad_dict, sp_dict, grads=True)

        global_step += 1

    N = batch_idx + 1
    train_loss = train_loss / N
    train_sparsity = train_sparsity / N
    predicted = np.asarray(predicted)
    labels = np.asarray(labels)
    train_acc = 100.0 * np.mean(predicted == labels)
    print('Epoch #{} (TRAIN): loss={}\tacc={}\tsparsity={}'
          .format(epoch + 1, train_loss, train_acc, train_sparsity))
    if args.debug:
        print('Average forward time over %d steps: %f' %(batch_idx, acc_forward_time / batch_idx))
        print('Average backward time over %d steps: %f' %(batch_idx, acc_backward_time / batch_idx))

    # validation
    net.eval()
    val_loss = 0
    val_loss_ce = 0
    val_loss_wd = 0
    val_loss_ad = 0
    val_sparsity = 0
    val_ad_dict = {}
    val_sp_dict = {}
    predicted = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss_ce = criterion(outputs['logits'], targets)
            loss_wd = weight_decay(net)
            loss_ad, ad_dict = activation_decay(outputs)
            sp_dict = activation_sparsity(outputs)
            loss = loss_ce + loss_wd + loss_ad

            val_loss    += loss.item()
            val_loss_ce += loss_ce.item()
            val_loss_wd += loss_wd.item()
            val_loss_ad += loss_ad.item()
            if args.debug:
                for key, val in ad_dict.items():
                    val_ad_dict[key] = val_ad_dict.get(key, 0) + val
                for key, val in sp_dict.items():
                    val_sp_dict[key] = val_sp_dict.get(key, 0) + val

            predicted.extend(outputs['logits'].max(1)[1].cpu().numpy())
            val_sparsity += calc_sparsity(sp_dict)

    N = batch_idx + 1
    val_loss    = val_loss / N
    val_loss_ce = val_loss_ce / N
    val_loss_wd = val_loss_wd / N
    val_loss_ad = val_loss_ad / N
    predicted = np.asarray(predicted)
    val_acc = 100.0 * np.mean(predicted == y_val)
    val_adv_acc, val_adv_robustness = calc_robustness(X_val, predicted, y_val, y_val_targets)
    val_sparsity = val_sparsity / N

    val_writer.add_scalar('losses/loss',    val_loss,    global_step)
    val_writer.add_scalar('losses/loss_ce', val_loss_ce, global_step)
    val_writer.add_scalar('losses/loss_wd', val_loss_wd, global_step)
    val_writer.add_scalar('losses/loss_ad', val_loss_ad, global_step)

    val_writer.add_scalar('metrics/acc', val_acc, global_step)
    val_writer.add_scalar('metrics/sparsity', val_sparsity, global_step)
    val_writer.add_scalar('metrics/adv_acc', val_adv_acc, global_step)
    val_writer.add_scalar('metrics/adv_robustness', val_adv_robustness, global_step)

    if args.debug:
        collect_debug(val_writer, val_ad_dict, val_sp_dict, N)

    if args.metric == 'accuracy':
        metric = val_acc
    elif args.metric == 'sparsity':
        metric = val_sparsity
    else:
        raise AssertionError('Unknown metric for optimization {}'.format(args.metric))

    if metric > best_metric:
        best_metric = metric
        print('Found new best model. Saving...')
        global_state['best_net'] = net.state_dict()
        global_state['best_metric'] = best_metric
        global_state['epoch'] = epoch
        global_state['global_step'] = global_step

    print('Epoch #{} (VAL): loss={}\tacc={:.2f}\tadv_acc={:.2f}\trobustness={:.4f}\tsparsity={:.4f}\tbest_metric({})={}'
          .format(epoch + 1, val_loss, val_acc, val_adv_acc, val_adv_robustness, val_sparsity, args.metric, best_metric))

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
        test_loss_wd = 0
        test_loss_ad = 0
        test_sparsity = 0
        test_ad_dict = {}
        test_sp_dict = {}
        predicted = []

        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss_ce = criterion(outputs['logits'], targets)
            loss_wd = weight_decay(net)
            loss_ad, ad_dict = activation_decay(outputs)
            sp_dict = activation_sparsity(outputs)
            loss = loss_ce + loss_wd + loss_ad

            test_loss    += loss.item()
            test_loss_ce += loss_ce.item()
            test_loss_wd += loss_wd.item()
            test_loss_ad += loss_ad.item()
            if args.debug:
                for key, val in ad_dict.items():
                    test_ad_dict[key] = test_ad_dict.get(key, 0) + val
                for key, val in sp_dict.items():
                    test_sp_dict[key] = test_sp_dict.get(key, 0) + val

            predicted.extend(outputs['logits'].max(1)[1].cpu().numpy())
            test_sparsity += calc_sparsity(sp_dict)

    N = batch_idx + 1
    test_loss    = test_loss / N
    test_loss_ce = test_loss_ce / N
    test_loss_wd = test_loss_wd / N
    test_loss_ad = test_loss_ad / N
    predicted = np.asarray(predicted)
    test_acc = 100.0 * np.mean(predicted == y_test)
    test_adv_acc, test_adv_robustness = calc_robustness(X_test, predicted, y_test, y_test_targets)
    test_sparsity = test_sparsity / N

    test_writer.add_scalar('losses/loss',    test_loss,    global_step)
    test_writer.add_scalar('losses/loss_ce', test_loss_ce, global_step)
    test_writer.add_scalar('losses/loss_wd', test_loss_wd, global_step)
    test_writer.add_scalar('losses/loss_ad', test_loss_ad, global_step)

    test_writer.add_scalar('metrics/acc', test_acc, global_step)
    test_writer.add_scalar('metrics/sparsity', test_sparsity, global_step)
    test_writer.add_scalar('metrics/adv_acc', test_adv_acc, global_step)
    test_writer.add_scalar('metrics/adv_robustness', test_adv_robustness, global_step)

    if args.debug:
        collect_debug(test_writer, test_ad_dict, test_sp_dict, N)

    print('Epoch #{} (TEST): loss={}\tacc={:.2f}\tadv_acc={:.2f}\trobustness={:.4f}\tsparsity={}'
          .format(epoch + 1, test_loss, test_acc, test_adv_acc, test_adv_robustness, test_sparsity))

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
        best_metric       = checkpoint['best_metric']
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
    classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=criterion,
                                   optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10)

    # create X_val and X_test which are normalized:
    X_val = -1.0 * np.ones(shape=(val_size, 3, 32, 32), dtype=np.float32)
    X_test = -1.0 * np.ones(shape=(test_size, 3, 32, 32), dtype=np.float32)
    for batch_idx, (inputs, targets) in enumerate(valloader):
        b = batch_idx * BATCH_SIZE
        e = b + targets.shape[0]
        X_val[b:e] = inputs.cpu().numpy()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        b = batch_idx * BATCH_SIZE
        e = b + targets.shape[0]
        X_test[b:e] = inputs.cpu().numpy()

    y_val = np.array(valloader.dataset.targets)
    y_test = np.asarray(testloader.dataset.targets)

    if args.targeted:
        if not os.path.isfile(os.path.join(ATTACK_DIR, 'y_val_targets.npy')):
            y_val_targets = random_targets(np.asarray(y_val), len(classes))  # .argmax(axis=1)
            np.save(os.path.join(ATTACK_DIR, 'y_val_targets.npy'), y_val_targets.argmax(axis=1))
            y_test_targets = random_targets(np.asarray(y_test), len(classes))  # .argmax(axis=1)
            np.save(os.path.join(ATTACK_DIR, 'y_test_targets.npy'), y_test_targets.argmax(axis=1))
        else:
            y_val_targets = np.load(os.path.join(ATTACK_DIR, 'y_val_targets.npy'))
            y_val_targets = to_categorical(y_val_targets, nb_classes=len(classes))
            y_test_targets = np.load(os.path.join(ATTACK_DIR, 'y_test_targets.npy'))
            y_test_targets = to_categorical(y_test_targets, nb_classes=len(classes))
    else:
        y_val_targets = None
        y_test_targets = None

    if args.attack == 'fgsm':
        attack = FastGradientMethod(
            classifier=classifier,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            targeted=args.targeted,
            num_random_init=0,
            batch_size=BATCH_SIZE
        )
    else:
        err_str = print('Attack {} is not supported'.format(args.attack))
        print(err_str)
        raise AssertionError(err_str)

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
    flush()

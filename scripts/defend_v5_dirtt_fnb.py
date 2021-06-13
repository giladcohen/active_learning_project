"""
This script uses TTA for robustness. I use the same "defend_v5_dirtt.py script, but instead of maintaining the
same btach for all the mini steps, I force new batch (FNB) for each iteration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
import torchvision.transforms as transforms

import logging
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import time
import sys
import PIL
from torchlars import LARS
import matplotlib.pyplot as plt

sys.path.insert(0, "..")

from active_learning_project.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101
import active_learning_project.datasets.my_transforms as my_transforms
from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor, \
    get_single_img_dataloader
from active_learning_project.utils import EMA, update_moving_average, convert_tensor_to_image
from active_learning_project.losses.vat import ConditionalEntropyLoss, kl_loss


parser = argparse.ArgumentParser(description='PyTorch TTA defense V4')
parser.add_argument('--checkpoint_dir',
                    default='/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00',
                    type=str, help='checkpoint dir')
parser.add_argument('--attack_dir', default='cw_targeted', type=str, help='attack directory')

# train
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
parser.add_argument('--steps', default=100, type=int, help='number of training steps')
parser.add_argument('--train_batch_size', default=100, type=int, help='batch size for the CLR training')
parser.add_argument('--ema_decay', default=0.998, type=float, help='EMA decay')

# loss
parser.add_argument('--lambda_cent', default=0.01, type=float, help='lambda_t in the paper')
parser.add_argument('--lambda_param', default=0.01, type=float, help='betha_t in the paper')
parser.add_argument('--lambda_vat', default=0.01, type=float, help='lambda_t in the paper, but I duplicate')
parser.add_argument('--n_power', default=1, type=int, help='VAT number of adversarial steps')
parser.add_argument('--xi', default=1e-6, type=float, help='VAT factor to multiply the adv perturbation noise')
parser.add_argument('--radius', default=3.5, type=float, help='VAT perturbation 2-norm ball radius')

# optimizer
parser.add_argument('--opt', default='adam', type=str, help='optimizer: sgd, adam, rmsprop, lars')
parser.add_argument('--mom', default=0.0, type=float, help='momentum of sgd optimizer')
parser.add_argument('--adam_b1', default=0.9, type=float, help='momentum of adam optimizer')

# eval
parser.add_argument('--tta_size', default=1000, type=int, help='number of test-time augmentations')
parser.add_argument('--eval_batch_size', default=100, type=int, help='batch size for the CLR training')
parser.add_argument('--mini_test', action='store_true', help='test only 2500 mini_test_inds')

# transforms:
parser.add_argument('--gaussian_std', default=0.0, type=float, help='Standard deviation of Gaussian noise') # was 0.0125

# debug:
parser.add_argument('--debug_size', default=None, type=int, help='number of image to run in debug mode')
parser.add_argument('--dump_dir', default='tmp', type=str, help='the dump dir')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()
args.mini_test = True

TRAIN_TIME_CNT = 0.0
TEST_TIME_CNT = 0.0
ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
DUMP_DIR = os.path.join(ATTACK_DIR, args.dump_dir)
os.makedirs(DUMP_DIR, exist_ok=True)

# dumping args to txt file
with open(os.path.join(DUMP_DIR, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

logging.basicConfig(filename=os.path.join(DUMP_DIR, 'log.log'),
                    filemode='w',
                    format='%(asctime)s %(name)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

# logger = logging.getLogger()
def log(str):
    logging.info(str)
    print(str)

def calc_robust_metrics(robustness_preds, robustness_preds_adv):
    acc_all = np.mean(robustness_preds[all_test_inds] == y_test[all_test_inds])
    acc_all_adv = np.mean(robustness_preds_adv[all_test_inds] == y_test[all_test_inds])
    return acc_all, acc_all_adv
    # log('Robust classification accuracy: all samples: {:.2f}/{:.2f}%'.format(acc_all * 100, acc_all_adv * 100))

def calc_first_n_robust_metrics(robustness_preds, robustness_preds_adv, n):
    acc_all = np.mean(robustness_preds[all_test_inds][0:n] == y_test[all_test_inds][0:n])
    acc_all_adv = np.mean(robustness_preds_adv[all_test_inds][0:n] == y_test[all_test_inds][0:n])
    return acc_all, acc_all_adv

def calc_first_n_robust_metrics_from_probs_summation(tta_robustness_probs, tta_robustness_probs_adv, n):
    tta_robustness_probs_sum = tta_robustness_probs.sum(axis=1)
    robustness_preds = tta_robustness_probs_sum.argmax(axis=1)
    tta_robustness_probs_adv_sum = tta_robustness_probs_adv.sum(axis=1)
    robustness_preds_adv = tta_robustness_probs_adv_sum.argmax(axis=1)
    acc_all = np.mean(robustness_preds[all_test_inds][0:n] == y_test[all_test_inds][0:n])
    acc_all_adv = np.mean(robustness_preds_adv[all_test_inds][0:n] == y_test[all_test_inds][0:n])
    return acc_all, acc_all_adv

def get_logits_from_emb_center(tta_embedding, model):
    tta_embeddings_center = tta_embedding.mean(axis=0)
    logits = model.linear(tta_embeddings_center)
    return logits

with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
args.dataset = train_args['dataset']

with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
    attack_args = json.load(f)
targeted = attack_args['targeted']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')

rand_gen = np.random.RandomState(12345)
train_normal_writer = SummaryWriter(os.path.join(DUMP_DIR, 'train_normal'))
train_adv_writer    = SummaryWriter(os.path.join(DUMP_DIR, 'train_adv'))
eval_normal_writer = SummaryWriter(os.path.join(DUMP_DIR, 'eval_normal'))
eval_adv_writer    = SummaryWriter(os.path.join(DUMP_DIR, 'eval_adv'))

def flush():
    train_normal_writer.flush()
    train_adv_writer.flush()
    eval_normal_writer.flush()
    eval_adv_writer.flush()

# Data
log('==> Preparing data..')
testloader = get_test_loader(
    dataset=train_args['dataset'],
    batch_size=args.eval_batch_size,
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
    my_transforms.GaussianNoise(0, args.gaussian_std),
    my_transforms.Clip(0.0, 1.0)
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

# summary(net, (3, 32, 32))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Model
if train_args['net'] == 'resnet18':
    model = ResNet18
elif train_args['net'] == 'resnet34':
    model = ResNet34
elif train_args['net'] == 'resnet50':
    model = ResNet50
elif train_args['net'] == 'resnet101':
    model = ResNet101
else:
    raise AssertionError("network {} is unknown".format(train_args['net']))

ema = EMA(args.ema_decay)
def reset_net():
    global net
    net = model(num_classes=len(classes), activation=train_args['activation'])
    net = net.to(device)
    net.load_state_dict(global_state['best_net'])
    ema.register(net)

def reset_opt():
    global optimizer
    if args.opt == 'sgd':
        optimizer = optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=args.mom,
            weight_decay=args.wd,
            nesterov=args.mom > 0)
    elif args.opt == 'adam':
        optimizer = optim.Adam(
            net.parameters(),
            lr=args.lr,
            betas=(args.adam_b1, 0.999),
            weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(
            net.parameters(),
            lr=args.lr,
            weight_decay=args.wd)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            net.parameters(),
            lr=args.lr,
            momentum=args.mom,
            weight_decay=args.wd)
    elif args.opt == 'lars':
        optimizer = optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=args.mom,
            weight_decay=args.wd,
            nesterov=args.mom > 0)
        optimizer = LARS(optimizer, trust_coef=args.lars_coeff, eps=args.lars_eps)
    else:
        raise AssertionError('optimizer {} is not expected'.format(args.opt))

reset_net()
reset_opt()

# test images inds:
mini_test_inds = np.load(os.path.join(args.checkpoint_dir, 'mini_test_inds.npy'))
if args.mini_test:
    all_test_inds = mini_test_inds
else:
    all_test_inds = np.arange(len(X_test))
if args.debug_size is not None:
    all_test_inds = all_test_inds[:args.debug_size]
img_cnt = len(all_test_inds)

robustness_preds            = -1 * np.ones(test_size, dtype=np.int32)
robustness_preds_adv        = -1 * np.ones(test_size, dtype=np.int32)

# multi TTAs
robustness_probs            = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)
robustness_probs_adv        = -1 * np.ones((test_size, args.tta_size, len(classes)), dtype=np.float32)

class VAT(nn.Module):
    def __init__(self, n_power, xi, radius):
        super(VAT, self).__init__()
        self.n_power = n_power
        self.XI = xi
        self.epsilon = radius

    def forward(self, X, logit):
        vat_loss = self.virtual_adversarial_loss(X, logit)
        return vat_loss

    def generate_virtual_adversarial_perturbation(self, x, logit):
        d = torch.randn_like(x, device='cuda')

        for _ in range(self.n_power):
            d = self.XI * self.get_normalized_vector(d).requires_grad_()
            logit_m = net(x + d)['logits']
            dist = kl_loss(logit, logit_m)
            grad = torch.autograd.grad(dist, [d])[0]
            d = grad.detach()

        return self.epsilon * self.get_normalized_vector(d)

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def virtual_adversarial_loss(self, x, logit):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)
        logit_p = logit.detach()
        logit_m = net(x + r_vadv)['logits']
        loss = kl_loss(logit_p, logit_m)
        return loss

cent = ConditionalEntropyLoss().to(device)
vat_loss = VAT(args.n_power, args.xi, args.radius)

def train(set):
    if set == 'normal':
        writer = train_normal_writer
    else:
        writer = train_adv_writer

    global TRAIN_TIME_CNT, loss_ent, loss_kl
    start_time = time.time()
    reset_net()
    reset_opt()
    net.train()

    tta_cnt = 0
    cnt = 0
    prev_logits = None
    for batch_idx, (inputs, targets) in enumerate(tta_loader):  # happens #steps times
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = net(inputs)
        loss_ent = cent(out['logits'])
        loss_vat = vat_loss(inputs, out['logits'])
        loss_kl = kl_loss(out['logits'], prev_logits) if prev_logits is not None else 0.0
        loss = args.lambda_cent * loss_ent + args.lambda_vat * loss_vat + args.lambda_param * loss_kl

        # collect for tensorboard:
        writer.add_scalar('img_ind_{}/losses/loss'.format(img_ind), loss, cnt)
        writer.add_scalar('img_ind_{}/losses/loss_ent'.format(img_ind), args.lambda_cent * loss_ent, cnt)
        writer.add_scalar('img_ind_{}/losses/loss_vat'.format(img_ind), args.lambda_vat * loss_vat, cnt)
        if prev_logits is not None:
            writer.add_scalar('img_ind_{}/losses/loss_kl_param'.format(img_ind), args.lambda_param * loss_kl, cnt)

        loss.backward()
        optimizer.step()
        ema(net)
        prev_logits = out['logits'].detach()
        cnt += 1
        tta_cnt += inputs.size(0)

    assert tta_cnt == args.train_batch_size * args.steps, 'at the end of the training cnt ({}) != samples({})'. \
        format(tta_cnt, args.train_batch_size * args.steps)

    TRAIN_TIME_CNT += time.time() - start_time

def eval(set):
    global TEST_TIME_CNT, net
    if set == 'normal':
        x = X_test
        rob_preds     = robustness_preds
        rob_probs     = robustness_probs
    else:
        x = X_test_adv
        rob_preds     = robustness_preds_adv
        rob_probs     = robustness_probs_adv

    start_time = time.time()
    net.eval()

    # eval_loss = 0.0
    # eval_loss_ent = 0.0
    # eval_loss_vat = 0.0

    with torch.no_grad():
        rob_preds[img_ind] = net(torch.from_numpy(np.expand_dims(x[img_ind], 0)).to(device))['preds'].squeeze().detach().cpu().numpy()
        tta_cnt = 0
        for batch_idx, (inputs, targets) in enumerate(tta_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            b = tta_cnt
            e = min(tta_cnt + len(inputs), args.tta_size)
            rob_probs[img_ind, b:e] = net(inputs)['probs'][0:(e-b)].detach().cpu().numpy()
            tta_cnt += e-b
            if tta_cnt >= args.tta_size:
                break
    assert tta_cnt == args.tta_size, 'tta_cnt={} must match args.tta_size'.format(tta_cnt)
    TEST_TIME_CNT += time.time() - start_time

for i in tqdm(range(img_cnt)):
    # for i in range(img_cnt):  # debug
    img_ind = all_test_inds[i]
    # normal
    tta_loader = get_single_img_dataloader(args.dataset, X_test, y_test, args.train_batch_size, args.steps * args.train_batch_size,
                                           pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    train('normal')

    tta_loader = get_single_img_dataloader(args.dataset, X_test, y_test, args.eval_batch_size, args.tta_size,
                                           pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    eval('normal')

    # adv
    tta_loader = get_single_img_dataloader(args.dataset, X_test_adv, y_test, args.train_batch_size, args.steps * args.train_batch_size,
                                           pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    train('adv')
    tta_loader = get_single_img_dataloader(args.dataset, X_test_adv, y_test, args.eval_batch_size, args.tta_size,
                                           pin_memory=device=='cuda', transform=tta_transforms, index=img_ind)
    eval('adv')

    acc_all, acc_all_adv = calc_first_n_robust_metrics(robustness_preds, robustness_preds_adv, i + 1)
    tta_acc_all, tta_acc_all_adv = calc_first_n_robust_metrics_from_probs_summation(robustness_probs, robustness_probs_adv, i + 1)

    log('accuracy on the fly after {} samples: original image: {:.2f}/{:.2f}%, TTAs: {:.2f}/{:.2f}%'
        .format(i + 1, acc_all * 100, acc_all_adv * 100, tta_acc_all * 100, tta_acc_all_adv * 100))
    flush()

average_test_time = TEST_TIME_CNT / (2 * img_cnt)
log('average eval time per sample: {} secs'.format(average_test_time))

log('done')
logging.shutdown()
exit(0)


# debug:
X_test_img     = convert_tensor_to_image(x)
plt.imshow(X_test_img[img_ind])
plt.show()

X_test_tta_img = convert_tensor_to_image(inputs.detach().cpu().numpy())
for t_ind in range(0, 5):
    plt.imshow(X_test_tta_img[t_ind])
    plt.show()
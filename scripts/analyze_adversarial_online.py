import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import json
import os
import argparse

from active_learning_project.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor
from active_learning_project.models.resnet import ResNet34, ResNet101
from active_learning_project.utils import convert_tensor_to_image

import matplotlib
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.saliency_map import SaliencyMapMethod
from art.attacks.evasion.carlini import CarliniL2Method
from art.attacks.evasion.elastic_net import ElasticNet

from art.classifiers import PyTorchClassifier
from cleverhans.utils import random_targets, to_categorical

checkpoint_dir = '/Users/giladcohen/data/gilad/logs/adv_robustness/cifar10/resnet34/resnet34_00'
attack_str = 'cw' # fgsm/pgd/jsma/pgd/deepfool/cw/ead
targeted = True
batch_size = 5
seed = 12345

rand_gen = np.random.RandomState(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
CHECKPOINT_PATH = os.path.join(checkpoint_dir, 'ckpt.pth')

print('==> Preparing data..')
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
testloader = get_test_loader(
    dataset=train_args['dataset'],
    batch_size=batch_size,
    num_workers=1,
    pin_memory=True
)

classes = testloader.dataset.classes

# attack:
print('==> Building model..')
if train_args['net'] == 'resnet34':
    net = ResNet34(num_classes=len(classes))
elif train_args['net'] == 'resnet101':
    net = ResNet101(num_classes=len(classes))
else:
    raise AssertionError("network {} is unknown".format(train_args['net']))
net = net.to(device)
# summary(net, (3, 32, 32))

if device == 'cuda':
    cudnn.benchmark = True
net.load_state_dict(global_state['best_net'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=train_args['lr'],
    momentum=train_args['mom'],
    weight_decay=0.0,  # train_args['wd'],
    nesterov=train_args['mom'] > 0)

X_test = get_normalized_tensor(testloader, batch_size)
y_test = testloader.dataset.targets

ids = np.sort(np.random.choice(len(X_test), batch_size, replace=False))
X_test = X_test[ids]
y_test = np.asarray(y_test)[ids]

net.eval()
classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=criterion,
                               optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(classes))

if targeted:
    y_test_targets = random_targets(np.asarray(y_test), len(classes))
    y_test_adv = y_test_targets.argmax(axis=1)
else:
    y_test_adv = None
    y_test_targets = None

if attack_str == 'fgsm':
    attack = FastGradientMethod(
        estimator=classifier,
        norm=np.inf,
        eps=0.01,
        eps_step=0.001,
        targeted=targeted,
        num_random_init=0,
        batch_size=batch_size
    )
elif attack_str == 'pgd':
    attack = ProjectedGradientDescent(
        estimator=classifier,
        norm=np.inf,
        eps=0.01,
        eps_step=0.003,
        targeted=targeted,
        batch_size=batch_size
    )
elif attack_str == 'deepfool':
    attack = DeepFool(
        classifier=classifier,
        max_iter=50,
        epsilon=0.02,
        nb_grads=len(classes),
        batch_size=batch_size
    )
elif attack_str == 'jsma':
    attack = SaliencyMapMethod(
        classifier=classifier,
        theta=1.0,
        gamma=0.01,
        batch_size=batch_size
    )
elif attack_str == 'cw':
    attack = CarliniL2Method(
        classifier=classifier,
        confidence=0.8,
        targeted=targeted,
        initial_const=0.1,
        batch_size=batch_size
    )
elif attack_str == 'ead':
    attack = ElasticNet(
        classifier=classifier,
        confidence=0.8,
        targeted=targeted,
        beta=0.01,  # EAD paper shows good results for L1
        batch_size=batch_size,
        decision_rule='L1'
    )
else:
    err_str = 'Attack {} is not supported'.format(attack_str)
    print(err_str)
    raise AssertionError(err_str)

X_test_adv = attack.generate(x=X_test, y=y_test_targets)
test_adv_logits = classifier.predict(X_test_adv, batch_size=batch_size)
y_test_adv_preds = np.argmax(test_adv_logits, axis=1)
test_adv_accuracy = np.sum(y_test_adv_preds == y_test) / batch_size
print('Accuracy on adversarial test examples: {}%'.format(test_adv_accuracy * 100))

X_test     = convert_tensor_to_image(X_test)
X_test_adv = convert_tensor_to_image(X_test_adv)

N = batch_size
ROWS = 2
fig = plt.figure(figsize=(N, ROWS))
for i in range(N):
    fig.add_subplot(ROWS, N, i+1)
    plt.imshow(X_test[i])
    plt.axis('off')
    fig.add_subplot(ROWS, N, i + N + 1)
    plt.imshow(X_test_adv[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

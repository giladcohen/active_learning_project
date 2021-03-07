"""Plotting the debug stats values after each steps of re-training the network with SimCLR"""

import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

ATTACK_DIR = '/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00/cw_targeted'
cross_entropy        = np.load(os.path.join(ATTACK_DIR, 'cross_entropy.npy'))
cross_entropy_adv    = np.load(os.path.join(ATTACK_DIR, 'cross_entropy_adv.npy'))
entropy              = np.load(os.path.join(ATTACK_DIR, 'entropy.npy'))
entropy_adv          = np.load(os.path.join(ATTACK_DIR, 'entropy_adv.npy'))
confidences          = np.load(os.path.join(ATTACK_DIR, 'confidences.npy'))
confidences_adv      = np.load(os.path.join(ATTACK_DIR, 'confidences_adv.npy'))
loss_contrastive     = np.load(os.path.join(ATTACK_DIR, 'loss_contrastive.npy'))
loss_contrastive_adv = np.load(os.path.join(ATTACK_DIR, 'loss_contrastive_adv.npy'))
loss_entropy         = np.load(os.path.join(ATTACK_DIR, 'loss_entropy.npy'))
loss_entropy_adv     = np.load(os.path.join(ATTACK_DIR, 'loss_entropy_adv.npy'))

N_imgs, N_steps = cross_entropy.shape
x = np.arange(N_steps)
for n in range(N_imgs):
    plt.close('all')
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    fig.suptitle('normal image #{}'.format(n), horizontalalignment='center', fontdict={'fontsize': 8})
    ax1.plot(x, cross_entropy[n])
    ax1.set_title('cross entropy', fontdict={'fontsize': 12})
    ax2.plot(x, entropy[n])
    ax2.set_title('entropy', fontdict={'fontsize': 12})
    ax3.plot(x, confidences[n])
    ax3.set_title('confidence', fontdict={'fontsize': 12})
    ax4.plot(x, loss_contrastive[n])
    ax4.set_title('contrastive loss', fontdict={'fontsize': 12})
    ax5.plot(x, loss_entropy[n])
    ax5.set_title('entropy loss', fontdict={'fontsize': 12})
    plt.tight_layout(h_pad=0.7)
    plt.show()

# plot mean debug:
    plt.close('all')
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    fig.suptitle('all normals', horizontalalignment='center', fontdict={'fontsize': 8})
    ax1.plot(x, cross_entropy.mean(axis=0))
    ax1.set_title('cross entropy', fontdict={'fontsize': 12})
    ax2.plot(x, entropy.mean(axis=0))
    ax2.set_title('entropy', fontdict={'fontsize': 12})
    ax3.plot(x, confidences.mean(axis=0))
    ax3.set_title('confidence', fontdict={'fontsize': 12})
    ax4.plot(x, loss_contrastive.mean(axis=0))
    ax4.set_title('contrastive loss', fontdict={'fontsize': 12})
    ax5.plot(x, loss_entropy.mean(axis=0))
    ax5.set_title('entropy loss', fontdict={'fontsize': 12})
    plt.tight_layout(h_pad=0.7)
    plt.show()

for n in range(N_imgs):
    plt.close('all')
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    fig.suptitle('adv image #{}'.format(n), horizontalalignment='center', fontdict={'fontsize': 8})
    ax1.plot(x, cross_entropy_adv[n], 'r')
    ax1.set_title('cross entropy', fontdict={'fontsize': 12})
    ax2.plot(x, entropy_adv[n], 'r')
    ax2.set_title('entropy', fontdict={'fontsize': 12})
    ax3.plot(x, confidences_adv[n], 'r')
    ax3.set_title('confidence', fontdict={'fontsize': 12})
    ax4.plot(x, loss_contrastive_adv[n], 'r')
    ax4.set_title('contrastive loss', fontdict={'fontsize': 12})
    ax5.plot(x, loss_entropy_adv[n], 'r')
    ax5.set_title('entropy loss', fontdict={'fontsize': 12})
    plt.tight_layout(h_pad=0.7)
    plt.show()

    # plot mean debug:
    plt.close('all')
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    fig.suptitle('all advs', horizontalalignment='center', fontdict={'fontsize': 8})
    ax1.plot(x, cross_entropy_adv.mean(axis=0), 'r')
    ax1.set_title('cross entropy', fontdict={'fontsize': 12})
    ax2.plot(x, entropy_adv.mean(axis=0), 'r')
    ax2.set_title('entropy', fontdict={'fontsize': 12})
    ax3.plot(x, confidences_adv.mean(axis=0), 'r')
    ax3.set_title('confidence', fontdict={'fontsize': 12})
    ax4.plot(x, loss_contrastive_adv.mean(axis=0), 'r')
    ax4.set_title('contrastive loss', fontdict={'fontsize': 12})
    ax5.plot(x, loss_entropy_adv.mean(axis=0), 'r')
    ax5.set_title('entropy loss', fontdict={'fontsize': 12})
    plt.tight_layout(h_pad=0.7)
    plt.show()

val_inds     = np.load(os.path.join(ATTACK_DIR, 'inds', 'val_inds.npy'))
f0_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f0_inds_val.npy'))
f1_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f1_inds_val.npy'))
f2_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f2_inds_val.npy'))
f3_inds_val  = np.load(os.path.join(ATTACK_DIR, 'inds', 'f3_inds_val.npy'))
test_inds    = np.load(os.path.join(ATTACK_DIR, 'inds', 'test_inds.npy'))
f0_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f0_inds_test.npy'))
f1_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f1_inds_test.npy'))
f2_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f2_inds_test.npy'))
f3_inds_test = np.load(os.path.join(ATTACK_DIR, 'inds', 'f3_inds_test.npy'))
y_test = np.load(os.path.join(os.path.dirname(os.path.join(ATTACK_DIR)), 'normal', 'y_test.npy'))
try:
    y_test_adv   = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))
except:
    print('nevermind')
y_test_preds     = np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy'))
y_test_adv_preds = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))



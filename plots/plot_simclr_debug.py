"""Plotting the debug stats values after each steps of re-training the network with SimCLR"""
from active_learning_project.utils import convert_tensor_to_image
from utils import majority_vote

NUM_DEBUG_SAMPLES = 200

import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, ".")
sys.path.insert(0, "./adversarial_robustness_toolbox")

ATTACK_DIR = '/data/gilad/logs/adv_robustness/cifar10/resnet34/regular/resnet34_00/cw_targeted'

# loading all stats:
robustness_preds           = np.load(os.path.join(ATTACK_DIR, 'robustness_preds.npy'))
robustness_preds_adv       = np.load(os.path.join(ATTACK_DIR, 'robustness_preds_adv.npy'))

# multi TTAs
robustness_probs           = np.load(os.path.join(ATTACK_DIR, 'robustness_probs.npy'))
robustness_probs_adv       = np.load(os.path.join(ATTACK_DIR, 'robustness_probs_adv.npy'))
robustness_probs_emb       = np.load(os.path.join(ATTACK_DIR, 'robustness_probs_emb.npy'))
robustness_probs_emb_adv   = np.load(os.path.join(ATTACK_DIR, 'robustness_probs_emb_adv.npy'))

# debug stats
# losses
loss_contrastive           = np.load(os.path.join(ATTACK_DIR, 'loss_contrastive.npy'))
loss_contrastive_adv       = np.load(os.path.join(ATTACK_DIR, 'loss_contrastive_adv.npy'))
loss_entropy               = np.load(os.path.join(ATTACK_DIR, 'loss_entropy.npy'))
loss_entropy_adv           = np.load(os.path.join(ATTACK_DIR, 'loss_entropy_adv.npy'))
loss_weight_difference     = np.load(os.path.join(ATTACK_DIR, 'loss_weight_difference.npy'))
loss_weight_difference_adv = np.load(os.path.join(ATTACK_DIR, 'loss_weight_difference_adv.npy'))
# per image stats
cross_entropy              = np.load(os.path.join(ATTACK_DIR, 'cross_entropy.npy'))
cross_entropy_adv          = np.load(os.path.join(ATTACK_DIR, 'cross_entropy_adv.npy'))
entropy                    = np.load(os.path.join(ATTACK_DIR, 'entropy.npy'))
entropy_adv                = np.load(os.path.join(ATTACK_DIR, 'entropy_adv.npy'))
confidences                = np.load(os.path.join(ATTACK_DIR, 'confidences.npy'))
confidences_adv            = np.load(os.path.join(ATTACK_DIR, 'confidences_adv.npy'))
# tta stats
tta_cross_entropy          = np.load(os.path.join(ATTACK_DIR, 'tta_cross_entropy.npy'))
tta_cross_entropy_adv      = np.load(os.path.join(ATTACK_DIR, 'tta_cross_entropy_adv.npy'))
tta_entropy                = np.load(os.path.join(ATTACK_DIR, 'tta_entropy.npy'))
tta_entropy_adv            = np.load(os.path.join(ATTACK_DIR, 'tta_entropy_adv.npy'))
tta_confidences            = np.load(os.path.join(ATTACK_DIR, 'tta_confidences.npy'))
tta_confidences_adv        = np.load(os.path.join(ATTACK_DIR, 'tta_confidences_adv.npy'))
tta_cross_entropy_emb      = np.load(os.path.join(ATTACK_DIR, 'tta_cross_entropy_emb.npy'))
tta_cross_entropy_emb_adv  = np.load(os.path.join(ATTACK_DIR, 'tta_cross_entropy_emb_adv.npy'))
tta_entropy_emb            = np.load(os.path.join(ATTACK_DIR, 'tta_entropy_emb.npy'))
tta_entropy_emb_adv        = np.load(os.path.join(ATTACK_DIR, 'tta_entropy_emb_adv.npy'))
tta_confidences_emb        = np.load(os.path.join(ATTACK_DIR, 'tta_confidences_emb.npy'))
tta_confidences_emb_adv    = np.load(os.path.join(ATTACK_DIR, 'tta_confidences_emb_adv.npy'))

y_test               = np.load(os.path.join(os.path.dirname(ATTACK_DIR), 'normal', 'y_test.npy'))
try:
    y_test_adv   = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))
except:
    print('nevermind')
y_test_preds     = np.load(os.path.join(ATTACK_DIR, 'y_test_preds.npy'))

X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
y_test_adv_preds = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))
X_test_adv_img = convert_tensor_to_image(X_test_adv)

# inds info:
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
f0_inds = list(f0_inds_val) + list(f0_inds_test)
f1_inds = list(f1_inds_val) + list(f1_inds_test)
f2_inds = list(f2_inds_val) + list(f2_inds_test)
f3_inds = list(f3_inds_val) + list(f3_inds_test)
f0_inds.sort()
f1_inds.sort()
f2_inds.sort()
f3_inds.sort()

# where did we misclassify after our re-training?
# get methods predictions: 1: simple, 2: majority vote, 3: softmax summation, 4: emb cluster softmax
preds = {}
preds_adv = {}
preds['simple']         = robustness_preds.copy()
preds_adv['simple']     = robustness_preds_adv.copy()
preds['majority']       = np.apply_along_axis(majority_vote, axis=1, arr=robustness_probs.argmax(axis=2))
preds_adv['majority']   = np.apply_along_axis(majority_vote, axis=1, arr=robustness_probs_adv.argmax(axis=2))
preds['summation']      = robustness_probs.sum(axis=1).argmax(axis=1)
preds_adv['summation']  = robustness_probs_adv.sum(axis=1).argmax(axis=1)
preds['emb_center']     = robustness_probs_emb.argmax(axis=1)
preds_adv['emb_center'] = robustness_probs_emb_adv.argmax(axis=1)

f0_robust_inds_normal = {}
f0_robust_inds_adv    = {}
f1_robust_inds_normal = {}
f1_robust_inds_adv    = {}

for key in preds.keys():
    f0_robust_inds_normal[key] = np.where(preds[key][:NUM_DEBUG_SAMPLES]         != y_test[:NUM_DEBUG_SAMPLES])[0]
    f0_robust_inds_adv[key]    = np.where(preds_adv[key][:NUM_DEBUG_SAMPLES]     != y_test[:NUM_DEBUG_SAMPLES])[0]
    f1_robust_inds_normal[key] = np.where(preds[key][:NUM_DEBUG_SAMPLES]         == y_test[:NUM_DEBUG_SAMPLES])[0]
    f1_robust_inds_adv[key]    = np.where(preds_adv[key][:NUM_DEBUG_SAMPLES]     == y_test[:NUM_DEBUG_SAMPLES])[0]

N_imgs, N_steps = cross_entropy.shape
x = np.arange(N_steps)
for n in range(N_imgs):
    plt.close('all')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(15, 15))
    fig.suptitle('normal image #{}'.format(n), horizontalalignment='center', fontdict={'fontsize': 8})
    ax1.plot(x, loss_contrastive[n])
    ax1.set_title('contrastive loss', fontdict={'fontsize': 12})
    ax2.plot(x, loss_entropy[n])
    ax2.set_title('entropy loss', fontdict={'fontsize': 12})
    ax3.plot(x, loss_weight_difference[n])
    ax3.set_title('weight diff loss', fontdict={'fontsize': 12})

    ax4.plot(x, cross_entropy[n])
    ax4.set_title('cross entropy [simple]', fontdict={'fontsize': 12})
    ax5.plot(x, tta_cross_entropy[n])
    ax5.set_title('cross entropy [summation]', fontdict={'fontsize': 12})
    ax6.plot(x, tta_cross_entropy_emb[n])
    ax6.set_title('cross entropy [emb center]', fontdict={'fontsize': 12})

    ax7.plot(x, entropy[n])
    ax7.set_title('entropy [simple]', fontdict={'fontsize': 12})
    ax8.plot(x, tta_entropy[n])
    ax8.set_title('entropy [summation]', fontdict={'fontsize': 12})
    ax9.plot(x, tta_entropy_emb[n])
    ax9.set_title('entropy [emb center]', fontdict={'fontsize': 12})

    ax10.plot(x, confidences[n])
    ax10.set_title('confidence [simple]', fontdict={'fontsize': 12})
    ax11.plot(x, tta_confidences[n])
    ax11.set_title('confidence [summation]', fontdict={'fontsize': 12})
    ax12.plot(x, tta_confidences_emb[n])
    ax12.set_title('confidence [emb center]', fontdict={'fontsize': 12})

    plt.tight_layout(h_pad=0.7)
    plt.show()

for n in range(N_imgs):
    plt.close('all')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(15, 15))
    fig.suptitle('adv image #{}'.format(n), horizontalalignment='center', fontdict={'fontsize': 8})
    ax1.plot(x, loss_contrastive_adv[n], 'r')
    ax1.set_title('contrastive loss', fontdict={'fontsize': 12})
    ax2.plot(x, loss_entropy_adv[n], 'r')
    ax2.set_title('entropy loss', fontdict={'fontsize': 12})
    ax3.plot(x, loss_weight_difference_adv[n], 'r')
    ax3.set_title('weight diff loss', fontdict={'fontsize': 12})

    ax4.plot(x, cross_entropy_adv[n], 'r')
    ax4.set_title('cross entropy [simple]', fontdict={'fontsize': 12})
    ax5.plot(x, tta_cross_entropy_adv[n], 'r')
    ax5.set_title('cross entropy [summation]', fontdict={'fontsize': 12})
    ax6.plot(x, tta_cross_entropy_emb_adv[n], 'r')
    ax6.set_title('cross entropy [emb center]', fontdict={'fontsize': 12})

    ax7.plot(x, entropy_adv[n], 'r')
    ax7.set_title('entropy [simple]', fontdict={'fontsize': 12})
    ax8.plot(x, tta_entropy_adv[n], 'r')
    ax8.set_title('entropy [summation]', fontdict={'fontsize': 12})
    ax9.plot(x, tta_entropy_emb_adv[n], 'r')
    ax9.set_title('entropy [emb center]', fontdict={'fontsize': 12})

    ax10.plot(x, confidences_adv[n], 'r')
    ax10.set_title('confidence [simple]', fontdict={'fontsize': 12})
    ax11.plot(x, tta_confidences_adv[n], 'r')
    ax11.set_title('confidence [summation]', fontdict={'fontsize': 12})
    ax12.plot(x, tta_confidences_emb_adv[n], 'r')
    ax12.set_title('confidence [emb center]', fontdict={'fontsize': 12})

    plt.tight_layout(h_pad=0.7)
    plt.show()

# # plot mean debug:
# plt.close('all')
# fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
# fig.suptitle('all normals', horizontalalignment='center', fontdict={'fontsize': 8})
# ax1.plot(x, cross_entropy.mean(axis=0))
# ax1.set_title('cross entropy', fontdict={'fontsize': 12})
# ax2.plot(x, entropy.mean(axis=0))
# ax2.set_title('entropy', fontdict={'fontsize': 12})
# ax3.plot(x, confidences.mean(axis=0))
# ax3.set_title('confidence', fontdict={'fontsize': 12})
# ax4.plot(x, loss_contrastive.mean(axis=0))
# ax4.set_title('contrastive loss', fontdict={'fontsize': 12})
# ax5.plot(x, loss_entropy.mean(axis=0))
# ax5.set_title('entropy loss', fontdict={'fontsize': 12})
# ax6.plot(x, loss_weight_difference.mean(axis=0))
# ax6.set_title('weight diff loss', fontdict={'fontsize': 12})
# plt.tight_layout(h_pad=0.7)
# plt.show()
#
# # plot mean debug:
# plt.close('all')
# fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
# fig.suptitle('all advs', horizontalalignment='center', fontdict={'fontsize': 8})
# ax1.plot(x, cross_entropy_adv.mean(axis=0), 'r')
# ax1.set_title('cross entropy', fontdict={'fontsize': 12})
# ax2.plot(x, entropy_adv.mean(axis=0), 'r')
# ax2.set_title('entropy', fontdict={'fontsize': 12})
# ax3.plot(x, confidences_adv.mean(axis=0), 'r')
# ax3.set_title('confidence', fontdict={'fontsize': 12})
# ax4.plot(x, loss_contrastive_adv.mean(axis=0), 'r')
# ax4.set_title('contrastive loss', fontdict={'fontsize': 12})
# ax5.plot(x, loss_entropy_adv.mean(axis=0), 'r')
# ax5.set_title('entropy loss', fontdict={'fontsize': 12})
# ax6.plot(x, loss_weight_difference_adv.mean(axis=0))
# ax6.set_title('weight diff loss', fontdict={'fontsize': 12})
# plt.tight_layout(h_pad=0.7)
# plt.show()
#
#

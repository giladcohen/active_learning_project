import matplotlib.pyplot as plt
import numpy as np

X_test     = np.load('/data/dynamic_wd/simple_wd_0.00039_mom_0.9_160220/fgsm_targeted/X_test_img.npy')
X_test_adv = np.load('/data/dynamic_wd/simple_wd_0.00039_mom_0.9_160220/fgsm_targeted/X_test_adv_img.npy')

plt.imshow(X_test[160])
plt.show()

plt.imshow(X_test_adv[160])
plt.show()

plt.imshow(X_test[161])
plt.show()

plt.imshow(X_test_adv[161])
plt.show()




import numpy as np
import torch
import torchvision.transforms as transforms
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
from art.attacks.attack import EvasionAttack

class TTAWhiteboxProjectedGradientDescent(ProjectedGradientDescentPyTorch):
    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "minimal",
        "max_iter",
        "random_eps",
    ]

    def __init__(
            self,
            estimator,
            norm: int = np.inf,
            eps: float = 0.3,
            eps_step: float = 0.1,
            max_iter: int = 100,
            targeted: bool = False,
            num_random_init: int = 0,
            batch_size: int = 32,
            random_eps: bool = False,
            tta_transforms: transforms.Compose = None,
            tta_size: int = 256):
        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps)

        self._check_params()
        self.tta_transforms = tta_transforms
        self.tta_size = tta_size

    def _check_params(self) -> None:
        # Check if order of the norm is acceptable given current implementation
        if self.norm not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either `np.inf`, 1, or 2.")

        if self.eps <= 0:
            raise ValueError("The perturbation size `eps` has to be positive.")

        if self.eps_step <= 0:
            raise ValueError("The perturbation step-size `eps_step` has to be positive.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if not isinstance(self.num_random_init, (int, np.int)):
            raise TypeError("The number of random initialisations has to be of type integer")

        if self.num_random_init < 0:
            raise ValueError("The number of random initialisations `random_init` has to be greater than or equal to 0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if self.eps_step > self.eps:
            raise ValueError("The iteration step `eps_step` has to be smaller than the total attack `eps`.")

        if self.max_iter <= 0:
            raise ValueError("The number of iterations `max_iter` has to be a positive integer.")

    def _compute_perturbation(self, x: "torch.Tensor", y: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
        """
        Compute perturbations.

        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :return: Perturbations.
        """
        import torch  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = np.nan * torch.ones(x.size(), device=x.device)
        for i in range(len(x)):
            grad[i] = self.estimator.tta_loss_gradient_framework(x[i], y[i], self.tta_transforms, self.tta_size) * (1 - 2 * int(self.targeted))

        # Apply norm bound
        if self.norm == np.inf:
            grad = grad.sign()

        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore

        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sqrt(torch.sum(grad * grad, axis=ind, keepdims=True)) + tol)  # type: ignore

        assert x.shape == grad.shape

        if mask is None:
            return grad
        else:
            return grad * mask

## debug
# import matplotlib.pyplot as plt
# from active_learning_project.utils import convert_tensor_to_image
# clipping
# x_clipped = torch.clip(img_ttas, 0.0, 1.0)
# x_img = convert_tensor_to_image(x_clipped.detach().cpu().numpy())
# for i in range(0, 5):
#     plt.imshow(x_img[i])
#     plt.show()

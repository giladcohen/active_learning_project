from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import torch

from active_learning_project.classifiers.pytorch_classifier_specific import PyTorchClassifierSpecific
from art.utils import CLIP_VALUES_TYPE
# from art.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)


class PyTorchTTAClassifier(PyTorchClassifierSpecific):  # lgtm [py/missing-call-to-init]

    def __init__(self,
                 model: "torch.nn.Module",
                 loss: "torch.nn.modules.loss._Loss",
                 input_shape: Tuple[int, ...],
                 nb_classes: int,
                 optimizer: Optional["torch.optim.Optimizer"] = None,  # type: ignore
                 clip_values: Optional[CLIP_VALUES_TYPE] = None,
                 fields=None,
                 ) -> None:
        super().__init__(
            model=model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            clip_values=clip_values,
            fields=fields
        )

    def tta_loss_gradient_framework(self, x: "torch.Tensor", y: "torch.Tensor", tta_transforms, tta_size) -> "torch.Tensor":
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_classes).
        :return: Gradients of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]
        from torch.autograd import Variable
        # torch.autograd.set_detect_anomaly(True)

        # Check label shape
        if self._reduce_labels:
            y = torch.argmax(y)

        y_ttas = y.repeat(tta_size)
        # create ttas variable:
        x_rep = x.repeat((tta_size, 1, 1, 1))
        x_rep = Variable(x_rep, requires_grad=True)

        x_ttas = np.nan * torch.ones(x_rep.size(), device=x.device)
        for i in range(tta_size):
            x_ttas[i] = tta_transforms(x_rep[i])
        assert not torch.isnan(x_ttas).any(), 'x_ttas must not have NaN values'

        # Compute the gradient and return
        model_outputs = self._model(x_ttas)
        loss = self._loss(model_outputs[-1], y_ttas)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = x_rep.grad
        assert grads.shape == x_rep.shape  # type: ignore
        assert not torch.isnan(grads).any(), 'Found Nan values in the TTAs grads'

        # average all grads:
        grads = torch.mean(grads, dim=0)

        return grads  # type: ignore

    def bpda_loss_gradient_framework(self, x: "torch.Tensor", y: "torch.Tensor", tta_transforms, tta_size) -> "torch.Tensor":
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_classes).
        :return: Gradients of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]
        from torch.autograd import Variable
        torch.autograd.set_detect_anomaly(True)

        # Check label shape
        if self._reduce_labels:
            y = torch.argmax(y)

        x.requires_grad = True
        # create ttas variable:
        x_rep = x.repeat((tta_size, 1, 1, 1))
        # x_rep = Variable(x_rep, requires_grad=True)

        x_ttas = np.nan * torch.ones(x_rep.size(), device=x.device)
        for i in range(tta_size):
            x_ttas[i] = tta_transforms(x_rep[i])
        assert not torch.isnan(x_ttas).any(), 'x_ttas must not have NaN values'

        # Compute the gradient and return
        model_outputs = self._model(x_ttas)
        loss = self._loss(model_outputs[-1], y.unsqueeze(0))

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = x.grad
        assert grads.shape == x.shape  # type: ignore
        assert not torch.isnan(grads).any(), 'Found Nan values in the TTAs grads'

        return grads  # type: ignore


# # debug
# import matplotlib.pyplot as plt
# from active_learning_project.utils import convert_tensor_to_image
# x_clipped = torch.clip(x_rep, 0.0, 1.0)
# x_img = convert_tensor_to_image(x_clipped.detach().cpu().numpy())
# for i in range(0, 5):
#     plt.imshow(x_img[i])
#     plt.show()
# x_grad = convert_tensor_to_image(grads.detach().cpu().numpy())
# for i in range(0, 5):
#     plt.imshow(x_grad[i])
#     plt.show()
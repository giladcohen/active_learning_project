from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
from torch.autograd import grad
import torch.nn.functional as F
import six

from art.config import ART_DATA_PATH, CLIP_VALUES_TYPE, PREPROCESSING_TYPE
from art.estimators.classification.classifier import (
    ClassGradientsMixin,
    ClassifierMixin,
)
from art.estimators.pytorch import PyTorchEstimator
from art.utils import Deprecated, deprecated_keyword_arg, check_and_transform_label_format

from active_learning_project.utils import jacobian, hessian, all_grads

if TYPE_CHECKING:
    from art.data_generators import DataGenerator
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)

from art.estimators.classification.pytorch import PyTorchClassifier

class PyTorchExtClassifier(PyTorchClassifier):  # lgtm [py/missing-call-to-init]

    def gradient_norm_gradient(self, x: np.ndarray, y: np.ndarray, out: str, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the norm of the gradient of out (pred/loss) w.r.t. `x`.

        :param x: Input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: Gradients of the same shape as `x`.
        """
        assert out in ['loss', 'pred'], 'out {} is not supported'.format(out)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Convert the inputs to Tensors
        inputs_t = torch.from_numpy(x_preprocessed).to(self._device)
        inputs_t.requires_grad = True

        # Convert the labels to Tensors
        labels_t = torch.from_numpy(y_preprocessed).to(self._device)

        # Compute the gradient and return
        model_outputs = self._model(inputs_t)
        if out == 'loss':
            # out_tensor = F.cross_entropy(model_outputs[-1]['logits'], labels_t, reduction='none')
            out_tensor = self._loss(model_outputs[-1]['logits'], labels_t)
            grad_outputs = None
        else:  # pred
            out_tensor = model_outputs[-1]['logits'].gather(1, labels_t.unsqueeze(1)).squeeze()
            grad_outputs = torch.tensor([1.0] * len(out_tensor)).to(self._device)

        # Clean gradients
        self._model.zero_grad()
        if inputs_t.grad is not None:
            inputs_t.grad.detach()
            inputs_t.grad.zero_()

        # Compute gradients
        # torch.autograd.backward(out_tensor, torch.tensor([1.0] * len(out_tensor)).to(self._device), create_graph=True)
        # norm_grad = torch.sum(torch.square(inputs_t.grad), dim=(1, 2, 3))
        # grads = all_grads(norm_grad, inputs_t, create_graph=False).detach().cpu().numpy()
        # grads = np.zeros((100, 3, 32, 32))

        # compute gradients try 2
        # img_grads = all_grads(out_tensor, inputs_t, create_graph=True)
        # norm_grad = torch.sum(torch.square(img_grads), dim=(1, 2, 3))
        # grads = all_grads(norm_grad, inputs_t, create_graph=False).detach().cpu().numpy()

        # compute gradients try 3
        k = 1e-15
        img_grads = torch.autograd.grad(out_tensor, inputs_t, grad_outputs=grad_outputs, create_graph=True)[0]
        # norm_grad = torch.square(img_grads), dim=(1, 2, 3)
        norm_grad = k * F.smooth_l1_loss(img_grads, torch.zeros_like(img_grads), reduce=False, beta=k)
        norm_grad = torch.sum(norm_grad, dim=(1, 2, 3))
        grads = torch.autograd.grad(norm_grad, inputs_t, grad_outputs=torch.tensor([1.0] * len(norm_grad)).to(self._device), create_graph=False)[0].detach().cpu().numpy()

        # grads = self._apply_preprocessing_gradient(x, norm_grad_grad)
        assert grads.shape == x.shape

        # cleaning:
        # norm_grad.detach()
        # del norm_grad
        # self._model.zero_grad()
        # # inputs_t.grad.detach()
        # del inputs_t

        return grads

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

    def __init__(self,
                 model: "torch.nn.Module",
                 loss: "torch.nn.modules.loss._Loss",
                 loss2: "torch.nn.modules.loss._Loss",
                 input_shape: Tuple[int, ...],
                 nb_classes: int,
                 optimizer: Optional["torch.optim.Optimizer"] = None,  # type: ignore
                 clip_values: Optional[CLIP_VALUES_TYPE] = None,
                 ) -> None:
        self._loss2 = loss2
        super(PyTorchExtClassifier, self).__init__(
             model=model,
             loss=loss,
             input_shape=input_shape,
             nb_classes=nb_classes,
             optimizer=optimizer,
             clip_values=clip_values
        )

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
        norm_grad = torch.square(img_grads)
        # norm_grad = k * F.smooth_l1_loss(img_grads, torch.zeros_like(img_grads), reduce=False, beta=k)
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


    def loss_preds_gradient_framework(self, x: "torch.Tensor", y: "torch.Tensor", wlg=False, wpg=False):
        """
        Compute the loss + preds + gradient of the loss + preds w.r.t. `x`.

        :param x: Input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param wlg: include losses gradients. Boolean.
        :param wpg: include preds gradients. Boolean.
        :return: losses + preds + Gradients of the same shape as `x` (for loss) or None + Gradients of the shape
        (B, N, cls, x.shape) (for preds) or None.
        """

        import torch  # lgtm [py/repeated-import]
        from torch.autograd import Variable

        # Check label shape
        if self._reduce_labels:
            y = torch.argmax(y, dim=1)

        # Convert the inputs to Variable
        x = Variable(x, requires_grad=True)

        # Compute the gradient and return
        model_outputs = self._model(x)
        preds = model_outputs[-1]['logits']
        loss_unreduced = self._loss2(model_outputs[-1]['logits'], y)

        if not (wlg or wpg):
            return loss_unreduced, preds

        # compute loss grads
        loss = loss_unreduced.mean()

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss_grads = torch.autograd.grad(loss, x, retain_graph=True)[0]
        assert loss_grads.shape == x.shape

        if wlg and not wpg:
            return loss_unreduced, preds, loss_grads

        # compute pred grads
        # clean gradients
        self._model.zero_grad()

        pred_grads = torch.empty((x.size(0), self.nb_classes) + x.size()[1:], dtype=torch.float32, device=self._device)
        for c in range(self.nb_classes):
            pred_grads[:, c] = torch.autograd.grad(preds[:, c], x, grad_outputs=torch.ones(preds.size(0), device=self._device), retain_graph=True)[0]

        return loss_unreduced, preds, loss_grads, pred_grads

    def loss_and_loss_gradient_framework(self, x: "torch.Tensor", y: "torch.Tensor", **kwargs) -> \
            Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: loss + Gradients of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]
        from torch.autograd import Variable

        # Check label shape
        if self._reduce_labels:
            y = torch.argmax(y, dim=1)

        # Convert the inputs to Variable
        x = Variable(x, requires_grad=True)

        # Compute the gradient and return
        model_outputs = self._model(x)
        loss = self._loss(model_outputs[-1]['logits'], y)
        loss_unreduced = self._loss2(model_outputs[-1]['logits'], y)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = x.grad
        assert grads.shape == x.shape  # type: ignore

        return loss_unreduced, grads  # type: ignore

    def preds_and_class_gradient(self, x: np.ndarray, label: Union[int, List[int], None] = None, **kwargs) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        import torch  # lgtm [py/repeated-import]

        if not (
            (label is None)
            or (isinstance(label, (int, np.integer)) and label in range(self._nb_classes))
            or (
                isinstance(label, np.ndarray)
                and len(label.shape) == 1
                and (label < self._nb_classes).all()
                and label.shape[0] == x.shape[0]
            )
        ):
            raise ValueError("Label %s is out of range." % label)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
        x_preprocessed = torch.from_numpy(x_preprocessed).to(self._device)

        # Compute gradients
        if self._layer_idx_gradients < 0:
            x_preprocessed.requires_grad = True

        # Run prediction
        model_outputs = self._model(x_preprocessed)

        # Set where to get gradient
        if self._layer_idx_gradients >= 0:
            input_grad = model_outputs[self._layer_idx_gradients]
        else:
            input_grad = x_preprocessed

        # Set where to get gradient from
        preds = model_outputs[-1]['logits']
        preds_np = preds.data.cpu().numpy()

        # Compute the gradient
        grads = []

        def save_grad():
            def hook(grad):
                grads.append(grad.cpu().numpy().copy())
                grad.data.zero_()

            return hook

        input_grad.register_hook(save_grad())

        self._model.zero_grad()
        if label is None:
            for i in range(self.nb_classes):
                torch.autograd.backward(
                    preds[:, i], torch.tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True,
                )

        elif isinstance(label, (int, np.integer)):
            torch.autograd.backward(
                preds[:, label], torch.tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True,
            )
        else:
            unique_label = list(np.unique(label))
            for i in unique_label:
                torch.autograd.backward(
                    preds[:, i], torch.tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True,
                )

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = grads[np.arange(len(grads)), lst]

            grads = grads[None, ...]

        grads = np.swapaxes(np.array(grads), 0, 1)
        grads = self._apply_preprocessing_gradient(x, grads)

        return preds_np, grads

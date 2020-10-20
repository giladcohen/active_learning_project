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

    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], None] = None, **kwargs) -> np.ndarray:
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

        # Compute the gradient
        grads = []
        grads_of_grads = []

        # def save_grad():
        #     def hook(grad):
        #         grad_of_grad = grad.grad
        #         print(grad.shape)
        #         print(grad[0, :, 0, 0])
        #
        #         print(grad_of_grad)
        #         print(grad_of_grad[0, :, 0, 0])
        #
        #         grads.append(grad.cpu().numpy().copy())
        #
        #         grad.data.zero_()
        #         print(grad[0, :, 0, 0])
        #         print(grad_of_grad[0, :, 0, 0])
        #
        #     return hook
        #
        # input_grad.register_hook(save_grad())

        self._model.zero_grad()
        if label is None:
            for i in range(self.nb_classes):
                torch.autograd.backward(
                    preds[:, i], torch.tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True, create_graph=True)

        elif isinstance(label, (int, np.integer)):
            torch.autograd.backward(
                preds[:, label], torch.tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True, create_graph=True)
        else:
            unique_label = list(np.unique(label))
            for i in unique_label:
                # for j in range(len(preds)):  # for each image
                    # img_grads = grad(preds[j, i], input_grad, create_graph=True)[0]
                    # print('cool')
                    # (img_grads_2,) = grad(img_grads[0,1,0,0], input_grad, create_graph=True) #, allow_unused=True)

                print('cool')
                # all_tensors = [input_grad] + list(self._model.parameters())
                # all_grads = grad(preds[0, i], all_tensors, create_graph=True)
                #
                # d2grad = []
                # for param, grd in zip(all_tensors, all_grads):
                #     for idx, _ in enumerate(param):
                #         drv = grad(grd[idx], param[idx], create_graph=True)
                #         d2grad.append(drv)
                #         print(param, drv)



                # x_grad_grad = grad(x_grad[0, 0, 0, 0], input_grad, create_graph=True)

                def func(x):
                    return self._model(x)[-1]['logits'][:, 0]

                # hassian_val = hessian(preds[:, i], input_grad)
                # # hessian_val = torch.autograd.functional.hessian(func, input_grad, create_graph=True)
                # jacobian_val = torch.autograd.functional.jacobian(func, input_grad, create_graph=True)
                # jacobian_trace = torch.zeros_like(input_grad)
                # for k in range(len(input_grad)):
                #     jacobian_trace[k] = jacobian_val[k, k]
                #
                # def sum_abs_func(x):
                #     return torch.sum(torch.square(x), dim=(1, 2, 3))
                #
                # sum_abs_jac = torch.autograd.functional.jacobian(sum_abs_func, input_grad, create_graph=False)
                # sum_abs_jac_trace = torch.zeros_like(input_grad)
                # for k in range(len(input_grad)):
                #     sum_abs_jac_trace[k] = sum_abs_jac[k, k]

                print('cool')
                # grad1 = all_grads(preds[:, i], input_grad, create_graph=True)
                # grad2 = all_grads(torch.sum(torch.square(grad1), dim=(1, 2, 3)), input_grad, create_graph=False)
                # grad3 = all_grads(grad1[:, 0, 0, 0], input_grad)

                preds_max = torch.max(preds, dim=1)[0]
                torch.autograd.backward(preds_max, torch.tensor([1.0] * len(preds_max)).to(self._device), create_graph=True)
                sum_abs_grad = torch.sum(torch.square(input_grad.grad), dim=(1, 2, 3))
                sub_abs_grad_grad = all_grads(sum_abs_grad, input_grad, create_graph=False)

                self._model.zero_grad()
                input_grad.grad.zero_()
                # torch.autograd.backward(mean_abs_grad, torch.tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True, create_graph=True)

                grad_y = torch.zeros_like(sum_abs_grad)
                grad_sum_abs_grad, = grad(sum_abs_grad, input_grad, grad_y, retain_graph=True, create_graph=True)


                # grads.append(input_grad.grad.cpu().detach().numpy().copy())

                # img_grads = grad(preds[:, i], input_grad, create_graph=True)
                # torch.autograd.backward(
                #     preds[:, i], torch.tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True, create_graph=True)

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = grads[np.arange(len(grads)), lst]

            grads = grads[None, ...]

        grads = np.swapaxes(np.array(grads), 0, 1)
        grads = self._apply_preprocessing_gradient(x, grads)

        return grads



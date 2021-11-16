from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import torch

from active_learning_project.classifiers.pytorch_classifier_specific import PyTorchClassifierSpecific
from art.utils import CLIP_VALUES_TYPE
# from art.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)


class SubstituteClassifier(PyTorchClassifierSpecific):  # lgtm [py/missing-call-to-init]

    def __init__(self,
                 model: "torch.nn.Module",
                 loss: "torch.nn.modules.loss._Loss",
                 input_shape: Tuple[int, ...],
                 nb_classes: int,
                 tta_size: int,
                 tta_transforms,
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
        self.tta_size = tta_size
        self.tta_transforms = tta_transforms

    def predict(  # pylint: disable=W0221
            self, x: np.ndarray, batch_size: int = 1, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        import torch  # lgtm [py/repeated-import]

        # Set model mode
        self._model.train(mode=training_mode)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        results_list = []

        for i in range(len(x_preprocessed)):
            x = torch.from_numpy(x_preprocessed[i]).to(self._device)
            x = x.unsqueeze(0)
            x_rep = x.repeat((self.tta_size, 1, 1, 1))
            x_ttas = np.nan * torch.ones(x_rep.size(), device=x.device)
            for i in range(self.tta_size):
                x_ttas[i] = self.tta_transforms(x_rep[i])
            assert not torch.isnan(x_ttas).any(), 'x_ttas must not have NaN values'

            model_outputs = self._model(x_ttas)
            output = model_outputs[-1]
            output = output.detach().cpu().numpy().astype(np.float32)
            if len(output.shape) == 1:
                output = np.expand_dims(output.detach().cpu().numpy(), axis=1).astype(np.float32)

            results_list.append(output)

        results = np.vstack(results_list)

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions

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
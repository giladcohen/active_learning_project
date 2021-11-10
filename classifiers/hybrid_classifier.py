import logging
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import torch
import torchvision
import sklearn
import matplotlib.pyplot as plt

from active_learning_project.classifiers.pytorch_classifier_specific import PyTorchClassifierSpecific
from active_learning_project.utils import convert_tensor_to_image
from active_learning_project.datasets.tta_utils import get_tta_logits

from art.utils import CLIP_VALUES_TYPE
# from art.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)


class HybridClassifier(PyTorchClassifierSpecific):  # lgtm [py/missing-call-to-init]

    def __init__(self,
                 dnn_model: torch.nn.Module,
                 rf_model: sklearn.ensemble.RandomForestClassifier,
                 dataset: str,
                 tta_args: Dict,
                 # tta_transforms: torchvision.transforms.Compose,
                 input_shape: Tuple[int, ...],
                 nb_classes: int,
                 clip_values: Optional[CLIP_VALUES_TYPE] = None,
                 fields=None
                 ) -> None:
        super().__init__(
            model=dnn_model,
            loss=None,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=None,
            clip_values=clip_values,
            fields=fields
        )
        self.rf_model = rf_model
        # self.tta_transforms = tta_transforms
        self.tta_args = tta_args
        self.dataset = dataset

    def predict(  # pylint: disable=W0221
            self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs
    ) -> np.ndarray:

        torch.autograd.set_detect_anomaly(True)
        self._model.train(mode=training_mode)

        y = -1 * np.ones(x.shape[0])
        tta_logits = get_tta_logits(self.dataset, self.tta_args, self._model._model, x, y, self.nb_classes)
        rf_features = tta_logits.reshape(x.shape[0], -1)  # (N, 2560)

        output_probs = self.rf_model.predict_proba(rf_features)
        return output_probs


    # def predict(  # pylint: disable=W0221
    #     self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs
    # ) -> np.ndarray:
    #
    #     torch.autograd.set_detect_anomaly(True)
    #     self._model.train(mode=training_mode)
    #
    #     # duplicate input to tta_size inputs
    #     x = torch.from_numpy(x).to(self.model.device)                # (N, 3, 32, 32)
    #     x_ttas = np.nan * torch.ones((x.size(0), self.tta_size) + self.input_shape, device=x.device)  # (N, 256, 3, 32, 32)
    #
    #     for k in range(x.size(0)):
    #         for i in range(self.tta_size):
    #             x_ttas[k, i] = self.tta_transforms(x[k])
    #     assert not torch.isnan(x_ttas).any(), 'x_ttas must not have NaN values'
    #
    #     rf_features = []
    #     with torch.no_grad():
    #         for k in range(x.size(0)):
    #             tmp_outputs = self._model(x_ttas[k])[-1].detach().cpu().numpy()  # (256, 10)
    #             rf_features.append(tmp_outputs)
    #     rf_features = np.stack(rf_features)  # (N, 256, 10)
    #     rf_features = rf_features.reshape(x.size(0), -1)  # (N, 2560)
    #
    #     output_probs = self.rf_model.predict_proba(rf_features)
    #     return output_probs

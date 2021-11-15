import logging
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import torch
import torchvision
import sklearn
import matplotlib.pyplot as plt
import argparse
import os
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
                 input_shape: Tuple[int, ...],
                 nb_classes: int,
                 clip_values: Optional[CLIP_VALUES_TYPE] = None,
                 fields=None,
                 tta_dir=None
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
        self.tta_args = tta_args
        self.dataset = dataset
        self.tta_dir = tta_dir

    def predict(  # pylint: disable=W0221
            self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs
    ) -> np.ndarray:

        torch.autograd.set_detect_anomaly(True)
        self._model.train(mode=training_mode)

        if self.tta_dir is not None:
            test_tta_file = os.path.join(self.tta_dir, 'tta_logits_test.npy')
            if os.path.exists(test_tta_file):
                logger.info('Loading test tta logits from {}'.format(test_tta_file))
                tta_logits = np.load(test_tta_file)
            else:
                logger.info('File {} is missing. Calculating test tta logits...'.format(test_tta_file))
                y = -1 * np.ones(x.shape[0])
                tta_logits = get_tta_logits(self.dataset, self._model._model, x, y, self.nb_classes, self.tta_args)
                logger.info('Saving test tta logits to {}'.format(test_tta_file))
                np.save(test_tta_file, tta_logits)
        else:
            logger.info('Calculating test tta logits...')
            y = -1 * np.ones(x.shape[0])
            tta_logits = get_tta_logits(self.dataset, self._model._model, x, y, self.nb_classes, self.tta_args)

        rf_features = tta_logits.reshape(x.shape[0], -1)  # (N, 2560)
        output_probs = self.rf_model.predict_proba(rf_features)
        return output_probs


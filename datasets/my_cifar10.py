from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import CIFAR10
import torch
from utils import inverse_map

class MyCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs) -> None:
        cls_to_omit = kwargs.pop('cls_to_omit', None)
        super(MyCIFAR10, self).__init__(*args, **kwargs)
        if cls_to_omit is not None:
            assert cls_to_omit in self.classes
            label_to_omit = self.class_to_idx[cls_to_omit]
            self.classes.remove(cls_to_omit)
            del self.class_to_idx[cls_to_omit]

            for cls_str, cls_label in self.class_to_idx.items():
                if cls_label < label_to_omit:
                    continue
                elif cls_label > label_to_omit:
                    self.class_to_idx[cls_str] = cls_label - 1
                else:
                    raise AssertionError('cls_label={} should have been deleted by now'.format(cls_label))

            indices_to_omit = np.where(np.asarray(self.targets) == label_to_omit)[0]
            mask = np.ones(len(self.data), dtype=bool)
            mask[indices_to_omit] = False
            self.data = self.data[mask]
            self.targets = np.asarray(self.targets)[mask].tolist()

            # update targets
            for i, target in enumerate(self.targets):
                if target < label_to_omit:
                    continue
                elif target > label_to_omit:
                    self.targets[i] = target - 1
                else:
                    raise AssertionError('target={} should have been deleted by now'.format(target))
        self.idx_to_class = inverse_map(self.class_to_idx)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if type(img) != torch.Tensor:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

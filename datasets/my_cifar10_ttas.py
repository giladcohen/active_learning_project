from typing import Any, Tuple
import torch
from torchvision.datasets import VisionDataset
import numpy as np

#debug:
import matplotlib.pyplot as plt
from active_learning_project.utils import convert_tensor_to_image

class TTADataset(VisionDataset):

    def __init__(self, data_norm, data_adv, tta_size, *args, **kwargs) -> None:
        root = None
        super().__init__(root, *args, **kwargs)
        self.data_norm = data_norm
        self.data_adv = data_adv
        self.tta_size = tta_size
        assert type(self.data_norm) == type(self.data_adv) == torch.Tensor, \
            'types of data_norm, data_adv must be tensor type'
        self.img_size = tuple(self.data_norm.size()[1:])
        self.full_tta_size = (tta_size, ) + self.img_size

    def __len__(self) -> int:
        return len(self.data_norm)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_norm, img_adv = self.data_norm[index], self.data_adv[index]

        # first, duplicate the image to TTAs:
        #img_norm = img_norm.repeat((self.tta_size, ) + (1, 1, 1))
        #img_adv = img_adv.repeat((self.tta_size, ) + (1, 1, 1))
        img_norm_ttas = torch.empty(self.full_tta_size)
        img_adv_ttas = torch.empty(self.full_tta_size)

        # now, transforming each image separately

        if self.transform is not None:
            for k in range(self.tta_size):
                img_norm_ttas[k] = self.transform(img_norm)
                img_adv_ttas[k] = self.transform(img_adv)

        x = torch.vstack((img_norm_ttas, img_adv_ttas))
        y = torch.tensor([0.0, 1.0])

        return x, y


from typing import Any, Tuple
import torch
from torchvision.datasets import VisionDataset

class TTADataset(VisionDataset):

    def __init__(self, data_norm, data_adv, tta_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        img_norm = img_norm.tile((self.tta_size, ) + (1, 1, 1))
        img_adv = img_adv.tile((self.tta_size, ) + (1, 1, 1))

        # now, transforming each image separately

        if self.transform is not None:
            img_norm = self.transform(img_norm)
            img_adv = self.transform(img_adv)

        x = torch.vstack((img_norm, img_adv))
        y = torch.tensor([0.0, 1.0])

        return x, y


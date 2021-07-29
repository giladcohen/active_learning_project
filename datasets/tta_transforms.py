import numpy as np
import PIL
import torchvision.transforms as transforms
import active_learning_project.datasets.my_transforms as my_transforms
from active_learning_project.utils import get_image_shape

def get_tta_transforms(dataset: str, gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = get_image_shape(dataset)
    n_pixels = img_shape[0]

    if clip_inputs:
        clip_min, clip_max = 0.0, 1.0
    else:
        clip_min, clip_max = -np.inf, np.inf

    if dataset in ['cifar10', 'cifar100', 'tiny_imagenet']:
        p_hflip = 0.5
    else:
        p_hflip = 0.0

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),  # To fix a bug where an ADV image has minus small value, applying gamma yields Nan
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  # padding to double the image size for rotation
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=32),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms

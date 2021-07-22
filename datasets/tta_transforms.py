import numpy as np
import PIL
import torchvision.transforms as transforms
import active_learning_project.datasets.my_transforms as my_transforms


def get_tta_transforms(dataset: str, gaussian_std: float=0.005, soft=False, clip_inputs=False):
    if clip_inputs:
        clip_min, clip_max = 0.0, 1.0
    else:
        clip_min, clip_max = -np.inf, np.inf
    p_hflip = 0.5 if 'cifar' in dataset else 0.0
    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),  # TO fix a bug where an ADV image has minus small value, applying gamma yields Nan
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=16, padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(4.0 / 64, 4.0 / 64),
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

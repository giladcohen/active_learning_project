import logging
import numpy as np
import torch
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from tqdm import tqdm
import PIL

from active_learning_project.datasets.train_val_test_data_loaders import dataset_factory
from active_learning_project.utils import convert_tensor_to_image
import active_learning_project.datasets.my_transforms as my_transforms
from art.utils import get_labels_np_array
# from art.utils import random_sphere
from typing import Union, Tuple
from scipy.special import gammainc
import matplotlib.pyplot as plt  # for debug

logger = logging.getLogger(__name__)

class TTABallExplorer(object):

    def __init__(
        self,
        classifier,
        dataset: str,
        rand_gen: np.random.RandomState,
        norm: int = np.inf,
        eps: float = 0.3,
        num_points: int = 10,
        batch_size: int = 32,
    ):
        self.classifier = classifier
        self.rand_gen = rand_gen
        self.norm = norm
        self.eps = eps
        self.num_points = num_points
        self.batch_size = batch_size
        self.dataset = dataset

    def generate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_tensor = torch.from_numpy(x.astype(np.float32))
        targets = get_labels_np_array(self.classifier.predict(x, batch_size=self.batch_size))
        num_classes = targets.shape[1]
        targets_sv = np.argmax(targets, axis=1)

        p_hflip = 0.5 if 'cifar' in self.dataset else 0.0
        tta_transforms = transforms.Compose([
            my_transforms.ColorJitterPro(
                brightness=[0.6, 1.4],
                contrast=[0.7, 1.3],
                saturation=[0.5, 1.5],
                hue=[-0.06, 0.06],
                gamma=[0.7, 1.3]
            ),
            transforms.Pad(padding=16, padding_mode='edge'),
            transforms.RandomAffine(
                degrees=15,
                translate=(4.0 / 64, 4.0 / 64),
                scale=(0.9, 1.1),
                shear=None,
                resample=PIL.Image.BILINEAR,
                fillcolor=None
            ),
            transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.5]),
            transforms.CenterCrop(size=32),
            transforms.RandomHorizontalFlip(p=p_hflip),
        ])

        data_dir, database, _, _ = dataset_factory(self.dataset)
        dataset     = database(root=data_dir, train=False, download=False, transform=None)
        tta_dataset = database(root=data_dir, train=False, download=False, transform=tta_transforms)

        # overwrite data in case of uint8 image

        # overwrite data in case of float32 image
        # x_img = np.transpose(x, (0, 2, 3, 1))
        # dataset.data = x_img
        # dataset.targets = targets.astype(np.float32)
        # tta_dataset.data = x_img
        # tta_dataset.targets = targets.astype(np.float32)

        # overwrite data in case of float32 input tensor
        dataset.data = x_tensor
        dataset.targets = targets.astype(np.float32)
        tta_dataset.data = x_tensor
        tta_dataset.targets = targets.astype(np.float32)

        # set loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
        tta_data_loader = torch.utils.data.DataLoader(
            tta_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        # all_x_adv = np.empty((x.shape[0], self.num_points) + x.shape[1:], dtype=np.float32)  # debug
        all_losses = np.empty((x.shape[0], self.num_points))
        all_preds = np.empty((x.shape[0], self.num_points, num_classes))
        all_noise_power = -1 * np.ones((x.shape[0], self.num_points))

        # The first sample (out of n samples) to be the original image, with original loss/preds
        for batch_id, batch in enumerate(data_loader):
            batch, batch_labels = batch
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = batch.to(self.classifier.device)
            batch_labels = batch_labels.to(self.classifier.device)
            losses, preds = self.classifier.loss_preds_gradient_framework(batch, batch_labels)

            # all_x_adv[batch_index_1:batch_index_2, 0]       = batch.data.cpu().numpy()  # debug
            all_losses[batch_index_1:batch_index_2, 0]      = losses.data.cpu().numpy()
            all_preds[batch_index_1:batch_index_2, 0]       = preds.data.cpu().numpy()
            all_noise_power[batch_index_1:batch_index_2, 0] = 0.0

        print('start predicting {} noisy samples...'.format(self.num_points - 1))
        for n in tqdm(range(1, self.num_points)):
            for batch_id, batch in enumerate(tta_data_loader):
                batch, batch_labels = batch
                batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
                out_np = self._generate_batch(batch, batch_labels)

                # all_x_adv[batch_index_1:batch_index_2, n]       = out_np[0]  # debug
                all_losses[batch_index_1:batch_index_2, n]      = out_np[1]
                all_preds[batch_index_1:batch_index_2, n]       = out_np[2]
                all_noise_power[batch_index_1:batch_index_2, n] = out_np[3]

        return _, all_losses, all_preds, all_noise_power

        # debug:
        # img = convert_tensor_to_image(all_x_adv[100*batch_id:100*(batch_id+1), 0])
        # img_adv = convert_tensor_to_image(batch.data.cpu().numpy())
        # img_adv_noisy = convert_tensor_to_image(all_x_adv[100*batch_id:100*(batch_id+1), 1])
        #
        # n_imgs = 10  # number of images
        # n_dist = 3  # number of distortions
        # inds = np.random.choice(np.arange(self.batch_size), n_imgs, replace=False)
        # fig = plt.figure(figsize=(n_dist, n_imgs))
        # for i in range(n_imgs):
        #     loc = 3 * i + 1
        #     fig.add_subplot(n_imgs, n_dist, loc)
        #     plt.imshow(img[inds[i]])
        #     plt.axis('off')
        #     loc = 3 * i + 2
        #     fig.add_subplot(n_imgs, n_dist, loc)
        #     plt.imshow(img_adv[inds[i]])
        #     plt.axis('off')
        #     loc = 3 * i + 3
        #     fig.add_subplot(n_imgs, n_dist, loc)
        #     plt.imshow(img_adv_noisy[inds[i]])
        #     plt.axis('off')
        # plt.tight_layout()
        # plt.show()

    def random_sphere(self, nb_points: int, nb_dims: int, max_radius: float, norm: Union[int, float]) -> np.ndarray:
        """
        Generate randomly `m x n`-dimension points with radius `radius` and centered around 0.

        :param nb_points: Number of random data points.
        :param nb_dims: Dimensionality of the sphere.
        :param radius: Maximum radius of the sphere.
        :param norm: Current support: 1, 2, np.inf.
        :param rand_gen: a random state
        :return: The generated random sphere.
        """
        radius_vec = self.rand_gen.uniform(0.0, float(max_radius), nb_points)
        res = np.empty((nb_points, nb_dims), dtype=np.float32)
        for i in range(nb_points):  # for every sample (with unique radius)...
            radius = radius_vec[i]
            if norm == 1:
                a_tmp = np.zeros(nb_dims + 1)
                a_tmp[-1] = radius
                a_tmp[1:-1] = np.sort(self.rand_gen.uniform(0, a_tmp[-1], nb_dims - 1))
                res[i] = (a_tmp[1:] - a_tmp[:-1]) * self.rand_gen.choice([-1, 1], nb_dims)
            elif norm == 2:
                a_tmp = self.rand_gen.randn(nb_dims)
                s_2 = np.sum(a_tmp ** 2)
                base = gammainc(nb_dims / 2.0, s_2 / 2.0) ** (1 / nb_dims) * radius / np.sqrt(s_2)
                res[i] = a_tmp * np.tile(base, nb_dims)
            elif norm == np.inf:
                res[i, :] = self.rand_gen.uniform(float(-radius), float(radius), nb_dims)

        return res

    def _generate_batch(self, x: "torch.Tensor", targets: "torch.Tensor") -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        x = x.to(self.classifier.device)
        targets = targets.to(self.classifier.device)
        num_classes = targets.shape[1]
        m = np.prod(x.shape[1:])

        random_perturbation = self.random_sphere(self.batch_size, m, self.eps, self.norm).astype(np.float32)
        noise_power = np.linalg.norm(random_perturbation, axis=1, ord=self.norm)
        random_perturbation = random_perturbation.reshape(x.shape)
        random_perturbation = torch.from_numpy(random_perturbation).to(self.classifier.device)

        x_adv = x + random_perturbation

        if self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

        losses_batch, preds_batch = self.classifier.loss_preds_gradient_framework(x_adv, targets)
        return x_adv.data.cpu().numpy(), losses_batch.data.cpu().numpy(), preds_batch.data.cpu().numpy(), noise_power

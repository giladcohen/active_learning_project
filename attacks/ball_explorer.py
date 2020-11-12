import logging
import numpy as np
import torch
from tqdm import tqdm

from art.utils import get_labels_np_array
# from art.utils import random_sphere
from typing import Union
from scipy.special import gammainc


logger = logging.getLogger(__name__)

class BallExplorer(object):

    def __init__(
        self,
        classifier,
        rand_gen: np.random.mtrand.RandomState,
        norm: int = np.inf,
        eps: float = 0.3,
        num_points: int = 10,
        batch_size: int = 32,
        output: str = 'loss'
    ):
        assert output in ['loss', 'pred']
        self.classifier = classifier
        self.rand_gen = rand_gen
        self.norm = norm
        self.eps = eps
        self.num_points = num_points
        self.batch_size = batch_size
        self.output = output

    def generate(self, x: np.ndarray) -> np.ndarray:

        targets = get_labels_np_array(self.classifier.predict(x, batch_size=self.batch_size))
        num_classes = targets.shape[1]
        targets_sv = np.argmax(targets, axis=1)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x.astype(np.float32)), torch.from_numpy(targets.astype(np.float32)),
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        if self.output == 'loss':
            all_grads = np.empty((x.shape[0], self.num_points) + x.shape[1:], dtype=np.float32)
        else:
            all_grads = np.empty((x.shape[0], self.num_points, num_classes) + x.shape[1:], dtype=np.float32)

        for batch_id, batch in tqdm(enumerate(data_loader)):
            batch, batch_labels = batch
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            all_grads[batch_index_1:batch_index_2] = self._generate_batch(batch, batch_labels)

        return all_grads

    def random_sphere(self, nb_points: int, nb_dims: int, radius: float, norm: Union[int, float]) -> np.ndarray:
        """
        Generate randomly `m x n`-dimension points with radius `radius` and centered around 0.

        :param nb_points: Number of random data points.
        :param nb_dims: Dimensionality of the sphere.
        :param radius: Radius of the sphere.
        :param norm: Current support: 1, 2, np.inf.
        :param rand_gen: a random state
        :return: The generated random sphere.
        """
        if norm == 1:
            a_tmp = np.zeros(shape=(nb_points, nb_dims + 1))
            a_tmp[:, -1] = np.sqrt(self.rand_gen.uniform(0, radius ** 2, nb_points))

            for i in range(nb_points):
                a_tmp[i, 1:-1] = np.sort(self.rand_gen.uniform(0, a_tmp[i, -1], nb_dims - 1))

            res = (a_tmp[:, 1:] - a_tmp[:, :-1]) * self.rand_gen.choice([-1, 1], (nb_points, nb_dims))
        elif norm == 2:
            a_tmp = self.rand_gen.randn(nb_points, nb_dims)
            s_2 = np.sum(a_tmp ** 2, axis=1)
            base = gammainc(nb_dims / 2.0, s_2 / 2.0) ** (1 / nb_dims) * radius / np.sqrt(s_2)
            res = a_tmp * (np.tile(base, (nb_dims, 1))).T
        elif norm == np.inf:
            res = self.rand_gen.uniform(float(-radius), float(radius), (nb_points, nb_dims))
        else:
            raise NotImplementedError("Norm {} not supported".format(norm))

        return res

    def _generate_batch(self, x: "torch.Tensor", targets: "torch.Tensor") -> np.ndarray:
        x = x.to(self.classifier.device)
        targets = targets.to(self.classifier.device)
        num_classes = targets.shape[1]

        n = self.num_points
        m = np.prod(x.shape)
        x_all = x.unsqueeze(dim=1).repeat(1, n, 1, 1, 1)  # copy x n times

        random_perturbation = self.random_sphere(n, m, self.eps, self.norm).reshape((x.shape[0], n) + x.shape[1:]).astype(np.float32)
        random_perturbation[:, 0] = np.zeros(x.shape)  # set zero perturbation for n=0 to get original grads.
        random_perturbation = torch.from_numpy(random_perturbation).to(self.classifier.device)

        x_adv = x_all + random_perturbation

        if self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

        # calculate grads on x_adv
        if self.output == 'loss':
            grads_batch = np.empty((self.batch_size, n) + x.shape[1:], dtype=np.float32)
            for i in range(self.num_points):
                grads_tensor = self.classifier.loss_gradient_framework(x_adv[:, i], targets)
                grads_batch[:, i] = grads_tensor.data.cpu().numpy()
        else:
            grads_batch = np.empty((self.batch_size, n, num_classes) + x.shape[1:], dtype=np.float32)
            for i in range(self.num_points):
                grads_batch[:, i] = self.classifier.class_gradient(x_adv[:, i].data.cpu().numpy(), label=None)

        return grads_batch

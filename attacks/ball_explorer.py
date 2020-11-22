import logging
import numpy as np
import torch
from tqdm import tqdm

from art.utils import get_labels_np_array
# from art.utils import random_sphere
from typing import Union, Tuple
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
        wlg: bool = True,
        wpg: bool = True
    ):
        self.classifier = classifier
        self.rand_gen = rand_gen
        self.norm = norm
        self.eps = eps
        self.num_points = num_points
        self.batch_size = batch_size
        self.wlg = wlg
        self.wpg = wpg

        assert not (self.wpg and not self.wlg)

    def generate(self, x: np.ndarray) -> \
            Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
                  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

        targets = get_labels_np_array(self.classifier.predict(x, batch_size=self.batch_size))
        num_classes = targets.shape[1]
        targets_sv = np.argmax(targets, axis=1)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x.astype(np.float32)), torch.from_numpy(targets.astype(np.float32)),
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        all_x_adv = np.empty((x.shape[0], self.num_points) + x.shape[1:], dtype=np.float32)

        all_losses = np.empty((x.shape[0], self.num_points))
        all_preds = np.empty((x.shape[0], self.num_points, num_classes))

        if self.wlg:
            all_losses_grads = np.empty((x.shape[0], self.num_points) + x.shape[1:], dtype=np.float32)
        if self.wpg:
            all_preds_grads = np.empty((x.shape[0], self.num_points, num_classes) + x.shape[1:], dtype=np.float32)

        for batch_id, batch in tqdm(enumerate(data_loader)):
            batch, batch_labels = batch
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            out_np = self._generate_batch(batch, batch_labels)
            all_x_adv[batch_index_1:batch_index_2]            = out_np[0]
            all_losses[batch_index_1:batch_index_2]           = out_np[1]
            all_preds[batch_index_1:batch_index_2]            = out_np[2]
            if self.wlg:
                all_losses_grads[batch_index_1:batch_index_2] = out_np[3]
            if self.wpg:
                all_preds_grads[batch_index_1:batch_index_2]  = out_np[4]

        if not (self.wlg or self.wpg):
            return all_x_adv, all_losses, all_preds
        elif self.wlg and not self.wpg:
            return all_x_adv, all_losses, all_preds, all_losses_grads
        else:
            return all_x_adv, all_losses, all_preds, all_losses_grads, all_preds_grads

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
        # radius_vec = np.tile(max_radius, nb_points)  # debug
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
            Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
                  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        x = x.to(self.classifier.device)
        targets = targets.to(self.classifier.device)
        num_classes = targets.shape[1]

        n = self.num_points
        m = np.prod(x.shape[1:])
        x_all = x.unsqueeze(dim=1).repeat(1, n, 1, 1, 1)  # copy x n times. shape = (b, n, 3, 32, 32)

        random_perturbation = self.random_sphere(n * self.batch_size, m, self.eps, self.norm).reshape((self.batch_size, n) + x.shape[1:]).astype(np.float32)
        random_perturbation[:, 0] = np.zeros(x.shape)  # set zero perturbation for n=0 to get original grads.
        random_perturbation = torch.from_numpy(random_perturbation).to(self.classifier.device)

        x_adv = x_all + random_perturbation

        if self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

        losses_batch        = np.empty((self.batch_size, n), dtype=np.float32)
        preds_batch         = np.empty((self.batch_size, n, num_classes), dtype=np.float32)
        if self.wlg:
            losses_grads_batch  = np.empty((self.batch_size, n) + x.shape[1:], dtype=np.float32)
        if self.wpg:
            preds_grads_batch   = np.empty((self.batch_size, n, num_classes) + x.shape[1:], dtype=np.float32)

        for i in range(self.num_points):
            out_tensor = self.classifier.loss_preds_gradient_framework(x_adv[:, i], targets, wlg=self.wlg, wpg=self.wpg)
            losses_batch[:, i] = out_tensor[0].data.cpu().numpy()
            preds_batch[:, i] = out_tensor[1].data.cpu().numpy()
            if self.wlg:
                losses_grads_batch[:, i] = out_tensor[2].data.cpu().numpy()
            if self.wpg:
                preds_grads_batch[:, i] = out_tensor[3].data.cpu().numpy()

        if not (self.wlg or self.wpg):
            return x_adv.data.cpu().numpy(), losses_batch, preds_batch
        elif self.wlg and not self.wpg:
            return x_adv.data.cpu().numpy(), losses_batch, preds_batch, losses_grads_batch
        else:
            return x_adv.data.cpu().numpy(), losses_batch, preds_batch, losses_grads_batch, preds_grads_batch

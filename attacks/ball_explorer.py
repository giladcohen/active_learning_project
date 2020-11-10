import logging
import numpy as np
import torch

from art.utils import get_labels_np_array
from art.utils import random_sphere


logger = logging.getLogger(__name__)

class BallExplorer(object):

    def __init__(
        self,
        classifier,
        norm: int = np.inf,
        eps: float = 0.3,
        num_points: int = 10,
        batch_size: int = 32,
        output: str = 'pred'
    ):
        assert output in ['loss', 'pred']
        self.classifier = classifier
        self.norm = norm
        self.eps = eps
        self.num_points = num_points
        self.batch_size = batch_size
        self.output = output

    def generate(self, x: np.ndarray) -> np.ndarray:

        targets = get_labels_np_array(self.classifier.predict(x, batch_size=self.batch_size))
        num_classes = targets.shape[1]
        targets_sv = np.argmax(targets)

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

        for batch_id, batch in enumerate(data_loader):
            batch, batch_labels = batch
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            all_grads[batch_index_1:batch_index_2] = self._generate_batch(batch, batch_labels)

        return all_grads

    def _generate_batch(self, x: "torch.Tensor", targets: "torch.Tensor") -> np.ndarray:
        x = x.to(self.classifier.device)
        targets = targets.to(self.classifier.device)
        num_classes = targets.shape[1]

        n = self.num_points
        m = np.prod(x.shape)
        x_all = x.unsqueeze(dim=1).repeat(1, n, 1, 1, 1)  # copy x n times

        random_perturbation = random_sphere(n, m, self.eps, self.norm).reshape((x.shape[0], n) + x.shape[1:]).astype(np.float32)
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



















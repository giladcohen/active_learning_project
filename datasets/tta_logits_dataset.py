import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split


class TTALogitsDataset(Dataset):
    def __init__(self, logits: torch.Tensor, rf_probs: torch.Tensor, y_gt: torch.Tensor):
        assert len(logits) == len(rf_probs)
        self.logits = logits
        self.rf_probs = rf_probs
        self.y_gt = y_gt

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, idx):
        logits = self.logits[idx]
        rf_probs = self.rf_probs[idx]
        y_gt = self.y_gt[idx]
        return logits, rf_probs, y_gt


# def get_logits_train_valid_loader(logits,
#                                   y_gt,
#                                   batch_size,
#                                   rand_gen,
#                                   valid_size,
#                                   shuffle=True,
#                                   num_workers=4,
#                                   pin_memory=False):
#
#     error_msg = "[!] valid_size should be in the range [0, 1]."
#     assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
#
#     num_train_val = len(logits)
#     num_val       = int(np.floor(valid_size * num_train_val))
#     indices = list(range(num_train_val))
#
#     train_idx, val_idx = \
#         train_test_split(indices, test_size=num_val, random_state=rand_gen, shuffle=shuffle, stratify=y_gt)

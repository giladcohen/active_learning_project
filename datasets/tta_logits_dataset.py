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

"""
Implementing MLP projection head similarly as in the SimCLR paper.
Ref: https://github.com/google-research/simclr/blob/master/model_util.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectobHead(nn.Module):
    def __init__(self, hidden_dim, mid_dim, out_dim):
        super(ProjectobHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(self.hidden_dim, self.mid_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.mid_dim)
        self.linear2 = nn.Linear(self.mid_dim, self.out_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(self.out_dim)

    def forward(self, x):
        # x = nn.functional.normalize(x)
        out = self.linear1(x)
        # out = self.bn1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)
        return out

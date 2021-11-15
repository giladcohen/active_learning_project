import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from active_learning_project.models import ResNet, MLP


class ResnetMlpStudent(nn.Module):
    """Substitute model the the Resnet + Random forest flow"""
    def __init__(self, resnet: ResNet, mlp: MLP):
        super().__init__()
        self.resnet = resnet
        self.mlp = mlp

    def forward(self, x):
        out_resnet = self.resnet(x)['logits']
        out_mlp = self.mlp(out_resnet)
        return out_mlp

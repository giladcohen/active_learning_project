import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """Multilayer Perceptron."""
    def __init__(self, num_classes, e=256):
        super().__init__()
        self.num_classes = num_classes
        self.e = e  # expansion

        if self.num_classes == 10:
            net_dims = num_classes * np.array([e, e,    e/2,  e/4,  e/8,  e/16, 1], dtype=int)
        else:
            net_dims = num_classes * np.array([e, e/10, e/20, e/40, e/80, e/160, 1], dtype=int)

        self.linear1 = nn.Linear(net_dims[0], net_dims[1], bias=False)
        self.bn1 = nn.BatchNorm1d(net_dims[1])

        self.linear2 = nn.Linear(net_dims[1], net_dims[2], bias=False)
        self.bn2 = nn.BatchNorm1d(net_dims[2])

        self.linear3 = nn.Linear(net_dims[2], net_dims[3], bias=False)
        self.bn3 = nn.BatchNorm1d(net_dims[3])

        self.linear4 = nn.Linear(net_dims[3], net_dims[4], bias=False)
        self.bn4 = nn.BatchNorm1d(net_dims[4])

        self.linear5 = nn.Linear(net_dims[4], net_dims[5], bias=False)
        self.bn5 = nn.BatchNorm1d(net_dims[5])

        self.linear6 = nn.Linear(net_dims[5], net_dims[6], bias=True)

    def forward(self, x):
        '''Forward pass'''
        net = {}
        out = x.view(x.size(0), -1)

        out = F.relu(self.bn1(self.linear1(out)))

        out = F.relu(self.bn2(self.linear2(out)))

        out = F.relu(self.bn3(self.linear3(out)))

        out = F.relu(self.bn4(self.linear4(out)))

        out = F.relu(self.bn5(self.linear5(out)))

        out = self.linear6(out)

        net['logits'] = out
        net['probs'] = F.softmax(out, dim=1)
        net['preds'] = net['probs'].argmax(dim=1)
        return net

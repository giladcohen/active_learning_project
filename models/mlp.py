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
        # net_dims = num_classes * np.array([e, e / 2, e / 4, e / 8, e / 16, 1], dtype=int)
        # net_dims = num_classes * np.array([e, e, e / 2, e / 4, e / 8, e / 16, 1], dtype=int)
        net_dims = num_classes * np.array([e, e, e, e/2, e/2, e/4, e/4, e/8, e/8, e/16, e/16, e/32, e/32, 1], dtype=int)

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

        self.linear6 = nn.Linear(net_dims[5], net_dims[6], bias=False)
        self.bn6 = nn.BatchNorm1d(net_dims[6])

        self.linear7 = nn.Linear(net_dims[6], net_dims[7], bias=False)
        self.bn7 = nn.BatchNorm1d(net_dims[7])

        self.linear8 = nn.Linear(net_dims[7], net_dims[8], bias=False)
        self.bn8 = nn.BatchNorm1d(net_dims[8])

        self.linear9 = nn.Linear(net_dims[8], net_dims[9], bias=False)
        self.bn9 = nn.BatchNorm1d(net_dims[9])

        self.linear10 = nn.Linear(net_dims[9], net_dims[10], bias=False)
        self.bn10 = nn.BatchNorm1d(net_dims[10])

        self.linear11 = nn.Linear(net_dims[10], net_dims[11], bias=False)
        self.bn11 = nn.BatchNorm1d(net_dims[11])

        self.linear12 = nn.Linear(net_dims[11], net_dims[12], bias=False)
        self.bn12 = nn.BatchNorm1d(net_dims[12])

        self.linear13 = nn.Linear(net_dims[12], net_dims[13], bias=True)

    def forward(self, x):
        '''Forward pass'''
        net = {}
        out = x.view(x.size(0), -1)

        out = F.relu(self.bn1(self.linear1(out)))
        out = F.relu(self.bn2(self.linear2(out)))
        out = F.relu(self.bn3(self.linear3(out)))
        out = F.relu(self.bn4(self.linear4(out)))
        out = F.relu(self.bn5(self.linear5(out)))
        out = F.relu(self.bn6(self.linear6(out)))
        out = F.relu(self.bn7(self.linear7(out)))
        out = F.relu(self.bn8(self.linear8(out)))
        out = F.relu(self.bn9(self.linear9(out)))
        out = F.relu(self.bn10(self.linear10(out)))
        out = F.relu(self.bn11(self.linear11(out)))
        out = F.relu(self.bn12(self.linear12(out)))
        out = self.linear13(out)

        net['logits'] = out
        net['probs'] = F.softmax(out, dim=1)
        net['preds'] = net['probs'].argmax(dim=1)
        return net

'''Network as implemented in: https://github.com/danieljakubovitz/Jacobian_Regularization'''
import torch.nn as nn
import torch.nn.functional as F

def to_1d(x):
    return x.view(x.size(0), -1)

class JakubovitzNet(nn.Module):
    def __init__(self, num_classes=10, return_logits_only=False):
        super(JakubovitzNet, self).__init__()
        self.dropout_rate = 0.0
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True)
        self.linear1 = nn.Linear(8 * 8 * 64, 1024, bias=True)
        self.linear2 = nn.Linear(1024, num_classes, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        net = {}
        net['images'] = x

        out = F.relu(self.conv1(x))


        out = F.max_pool2d(out, 2, padding=1)
        out = F.relu(self.conv2(out))

        out = F.max_pool2d(out, 2, padding=1)
        out = to_1d(out)
        out = F.relu(self.linear1(out))

        out = self.dropout(out)
        out = self.linear2(out)
        return out

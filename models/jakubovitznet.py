'''Network as implemented in: https://github.com/danieljakubovitz/Jacobian_Regularization'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_1d(x):
    return x.view(x.size(0), -1)

def activation_ratio(x):
    """
    :param x: feature map. tensor of size: [batch, feature_map_size, num_pix, num_pix], where num_pix=32/16/8/4
    :return: activation ratio per 2D conv kernel. size to return value: [batch, feature_map_size]
    """
    batch_size = x.size(0)
    is_1d = len(x.size()) == 2
    if is_1d:
        spatial_size = 1
        dim = 0
    else:
        spatial_size = x.size(2) * x.size(3)
        dim = (0, 2, 3)
    activated_sum = x.sign().sum(dim=dim)
    return activated_sum / (batch_size * spatial_size)

class JakubovitzNet(nn.Module):
    def __init__(self, num_classes=10, return_logits_only=False):
        super(JakubovitzNet, self).__init__()
        self.dropout_rate = 0.0
        self.return_logits_only = return_logits_only

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True)
        self.linear1 = nn.Linear(8 * 8 * 64, 1024, bias=True)
        self.linear2 = nn.Linear(1024, num_classes, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Weight that should be considered for the regularization on each ReLU activation
        self.weight_reg_dict = {
            'num_act1': ['conv1.weight', 'conv1.bias'],
            'num_act2': ['conv2.weight', 'conv2.bias'],
            'num_act3': ['linear1.weight', 'linear1.bias']
        }

    def forward(self, x):
        net = {}
        net['images'] = x

        out = F.relu(self.conv1(x))
        net['num_act1'] = out

        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        net['num_act2'] = out

        out = F.max_pool2d(out, 2)
        out = to_1d(out)
        out = F.relu(self.linear1(out))
        net['num_act3'] = out

        out = self.dropout(out)
        out = self.linear2(out)
        net['logits'] = out

        if self.return_logits_only:
            return net['logits']
        else:
            return net

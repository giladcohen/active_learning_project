import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def to_1d(x):
    return x.view(x.size(0), -1)

def activation_ratio(x):
    """
    :param x: feature map. tensor of size: [batch, feature_map_size, num_pix, num_pix], where num_pix=32/16/8/4
    :return: activation ratio per 2D conv kernel. size to return value: [batch, feature_map_size]
    """
    batch_size = x.size()[0]
    spatial_size = x.size()[2] * x.size()[3]
    activated_sum = x.sign().sum(dim=(0, 2, 3))
    return activated_sum / (batch_size * spatial_size)

class res_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, use_bn=True):
        super(res_basic, self).__init__()
        self.use_bn = use_bn
        self.use_bias = False  # not self.use_bn

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=self.use_bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=self.use_bias),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out1 = F.relu(out)

        out = self.dropout(self.conv2(out1))
        if self.use_bn:
            out = self.bn2(out)
        out += self.shortcut(x)
        out2 = F.relu(out)
        return out1, out2

        # old forward format
        # out = x
        # if self.use_bn:
        #     out = self.bn1(out)
        # out = F.relu(out)
        # relu_out1 = out.clone()
        #
        # out = self.dropout(self.conv1(out))
        # if self.use_bn:
        #     out = self.bn2(out)
        # out = F.relu(out)
        # relu_out2 = out.clone()
        #
        # out = self.conv2(out)
        # out += self.shortcut(x)
        #
        # return out, relu_out1, relu_out2


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # self.depth = 18
        self.dropout_rate = 0.0
        self.use_bn = True
        self.use_bias = False  # not self.use_bn

        self.nStages = [64, 64, 128, 256, 512]

        self.conv1 = nn.Conv2d(3, self.nStages[0], kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.bn1 = nn.BatchNorm2d(self.nStages[0])

        self.layer1_0 = res_basic(self.nStages[0], self.nStages[1], self.dropout_rate, stride=1, use_bn=self.use_bn)
        self.layer1_1 = res_basic(self.nStages[1], self.nStages[1], self.dropout_rate, stride=1, use_bn=self.use_bn)

        self.layer2_0 = res_basic(self.nStages[1], self.nStages[2], self.dropout_rate, stride=2, use_bn=self.use_bn)
        self.layer2_1 = res_basic(self.nStages[2], self.nStages[2], self.dropout_rate, stride=1, use_bn=self.use_bn)

        self.layer3_0 = res_basic(self.nStages[2], self.nStages[3], self.dropout_rate, stride=2, use_bn=self.use_bn)
        self.layer3_1 = res_basic(self.nStages[3], self.nStages[3], self.dropout_rate, stride=1, use_bn=self.use_bn)

        self.layer4_0 = res_basic(self.nStages[3], self.nStages[4], self.dropout_rate, stride=2, use_bn=self.use_bn)
        self.layer4_1 = res_basic(self.nStages[4], self.nStages[4], self.dropout_rate, stride=1, use_bn=self.use_bn)

        self.linear = nn.Linear(self.nStages[4], num_classes)

        # Weight that should be considered for the regularization on each ReLU activation
        self.weight_reg_dict = {
            'num_act1': ['conv1.weight', 'bn1.weight', 'bn1.bias'],
            'num_act2': ['layer1_0.conv1.weight', 'layer1_0.bn1.weight', 'layer1_0.bn1.bias'],
            'num_act3': ['layer1_0.conv2.weight', 'layer1_0.bn2.weight', 'layer1_0.bn2.bias'],
            'num_act4': ['layer1_1.conv1.weight', 'layer1_1.bn1.weight', 'layer1_1.bn1.bias'],
            'num_act5': ['layer1_1.conv2.weight', 'layer1_1.bn2.weight', 'layer1_1.bn2.bias'],
            'num_act6': ['layer2_0.conv1.weight', 'layer2_0.bn1.weight', 'layer2_0.bn1.bias'],
            'num_act7': ['layer2_0.conv2.weight', 'layer2_0.bn2.weight', 'layer2_0.bn2.bias',
                         'layer2_0.shortcut.0.weight', 'layer2_0.shortcut.1.weight', 'layer2_0.shortcut.1.bias'],
            'num_act8': ['layer2_1.conv1.weight', 'layer2_1.bn1.weight', 'layer2_1.bn1.bias'],
            'num_act9': ['layer2_1.conv2.weight', 'layer2_1.bn2.weight', 'layer2_1.bn2.bias'],
            'num_act10': ['layer3_0.conv1.weight', 'layer3_0.bn1.weight', 'layer3_0.bn1.bias'],
            'num_act11': ['layer3_0.conv2.weight', 'layer3_0.bn2.weight', 'layer3_0.bn2.bias',
                          'layer3_0.shortcut.0.weight', 'layer3_0.shortcut.1.weight', 'layer3_0.shortcut.1.bias'],
            'num_act12': ['layer3_1.conv1.weight', 'layer3_1.bn1.weight', 'layer3_1.bn1.bias'],
            'num_act13': ['layer3_1.conv2.weight', 'layer3_1.bn2.weight', 'layer3_1.bn2.bias'],
            'num_act14': ['layer4_0.conv1.weight', 'layer4_0.bn1.weight', 'layer4_0.bn1.bias'],
            'num_act15': ['layer4_0.conv2.weight', 'layer4_0.bn2.weight', 'layer4_0.bn2.bias',
                          'layer4_0.shortcut.0.weight', 'layer4_0.shortcut.1.weight', 'layer4_0.shortcut.1.bias'],
            'num_act16': ['layer4_1.conv1.weight', 'layer4_1.bn1.weight', 'layer4_1.bn1.bias'],
            'num_act17': ['layer4_1.conv2.weight', 'layer4_1.bn2.weight', 'layer4_1.bn2.bias']
        }

    def forward(self, x):
        net = {}

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = F.relu(out)
        net['num_act1'] = activation_ratio(out)

        relu_out, out = self.layer1_0(out)
        net['num_act2'] = activation_ratio(relu_out)
        net['num_act3'] = activation_ratio(out)
        relu_out, out = self.layer1_1(out)
        net['num_act4'] = activation_ratio(relu_out)
        net['num_act5'] = activation_ratio(out)

        relu_out, out = self.layer2_0(out)
        net['num_act6'] = activation_ratio(relu_out)
        net['num_act7'] = activation_ratio(out)
        relu_out, out = self.layer2_1(out)
        net['num_act8'] = activation_ratio(relu_out)
        net['num_act9'] = activation_ratio(out)

        relu_out, out = self.layer3_0(out)
        net['num_act10'] = activation_ratio(relu_out)
        net['num_act11'] = activation_ratio(out)
        relu_out, out = self.layer3_1(out)
        net['num_act12'] = activation_ratio(relu_out)
        net['num_act13'] = activation_ratio(out)

        relu_out, out = self.layer4_0(out)
        net['num_act14'] = activation_ratio(relu_out)
        net['num_act15'] = activation_ratio(out)
        relu_out, out = self.layer4_1(out)
        net['num_act16'] = activation_ratio(relu_out)
        net['num_act17'] = activation_ratio(out)

        out = F.avg_pool2d(out, 4)
        out = to_1d(out)
        net['embeddings'] = out
        out = self.linear(out)
        net['logits'] = out

        return net


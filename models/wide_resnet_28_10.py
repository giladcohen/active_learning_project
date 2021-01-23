import torch
import torch.nn as nn
import torch.nn.functional as F
from active_learning_project.utils import to_1d, activation_batch_ratio, activation_ratio_avg, activation_L1_ratio
from collections import OrderedDict

class res_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, use_bn=True, activation=F.relu):
        super(res_basic, self).__init__()
        self.activation = activation
        self.use_bn = use_bn
        self.use_bias = False  # not self.use_bn

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=self.use_bias)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if self.use_bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=self.use_bias),
                    nn.BatchNorm2d(planes))
            else:
                self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=self.use_bias))

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out1 = self.activation(out)

        out = self.dropout(self.conv2(out1))
        if self.use_bn:
            out = self.bn2(out)
        out += self.shortcut(x)
        out2 = self.activation(out)
        return out1, out2

class WideResNet28_10(nn.Module):
    def __init__(self, num_classes=10, use_bn=True, activation='relu'):
        super(WideResNet28_10, self).__init__()
        self.dropout_rate = 0.0
        self.use_bn = use_bn
        self.use_bias = False  # not self.use_bn
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'softplus':
            self.activation = F.softplus
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise AssertionError('activation function {} was not expected'.format(activation))

        self.nStages = [16, 160, 320, 640]

        # init_conv
        self.conv1 = nn.Conv2d(3, self.nStages[0], kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(self.nStages[0])

        # unit_1
        self.layer1_0 = res_basic(self.nStages[0], self.nStages[1], self.dropout_rate, stride=1, use_bn=self.use_bn, activation=self.activation)
        self.layer1_1 = res_basic(self.nStages[1], self.nStages[1], self.dropout_rate, stride=1, use_bn=self.use_bn, activation=self.activation)
        self.layer1_2 = res_basic(self.nStages[1], self.nStages[1], self.dropout_rate, stride=1, use_bn=self.use_bn, activation=self.activation)
        self.layer1_3 = res_basic(self.nStages[1], self.nStages[1], self.dropout_rate, stride=1, use_bn=self.use_bn, activation=self.activation)

        # unit_2
        self.layer2_0 = res_basic(self.nStages[1], self.nStages[2], self.dropout_rate, stride=2, use_bn=self.use_bn, activation=self.activation)
        self.layer2_1 = res_basic(self.nStages[2], self.nStages[2], self.dropout_rate, stride=1, use_bn=self.use_bn, activation=self.activation)
        self.layer2_2 = res_basic(self.nStages[2], self.nStages[2], self.dropout_rate, stride=1, use_bn=self.use_bn, activation=self.activation)
        self.layer2_3 = res_basic(self.nStages[2], self.nStages[2], self.dropout_rate, stride=1, use_bn=self.use_bn, activation=self.activation)

        # unit_3
        self.layer3_0 = res_basic(self.nStages[2], self.nStages[3], self.dropout_rate, stride=2, use_bn=self.use_bn, activation=self.activation)
        self.layer3_1 = res_basic(self.nStages[3], self.nStages[3], self.dropout_rate, stride=1, use_bn=self.use_bn, activation=self.activation)
        self.layer3_2 = res_basic(self.nStages[3], self.nStages[3], self.dropout_rate, stride=1, use_bn=self.use_bn, activation=self.activation)
        self.layer3_3 = res_basic(self.nStages[3], self.nStages[3], self.dropout_rate, stride=1, use_bn=self.use_bn, activation=self.activation)

        self.linear = nn.Linear(self.nStages[3], num_classes)

    def forward(self, x):
        net = OrderedDict()
        with torch.no_grad():
            net['images'] = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.activation(out)
        # with torch.no_grad():
        #     net['layer1'] = out

        relu_out, out = self.layer1_0(out)
        # with torch.no_grad():
        #     net['layer2'], net['layer3'] = relu_out, out
        relu_out, out = self.layer1_1(out)
        # with torch.no_grad():
        #     net['layer4'], net['layer5'] = relu_out, out
        relu_out, out = self.layer1_2(out)
        # with torch.no_grad():
        #     net['layer6'], net['layer7'] = relu_out, out
        relu_out, out = self.layer1_3(out)
        # with torch.no_grad():
        #     net['layer8'], net['layer9'] = relu_out, out

        relu_out, out = self.layer2_0(out)
        # with torch.no_grad():
        #     net['layer10'], net['layer11'] = relu_out, out
        relu_out, out = self.layer2_1(out)
        # with torch.no_grad():
        #     net['layer12'], net['layer13'] = relu_out, out
        relu_out, out = self.layer2_2(out)
        # with torch.no_grad():
        #     net['layer14'], net['layer15'] = relu_out, out
        relu_out, out = self.layer2_3(out)
        # with torch.no_grad():
        #     net['layer16'], net['layer17'] = relu_out, out

        relu_out, out = self.layer3_0(out)
        # with torch.no_grad():
        #     net['layer18'], net['layer19'] = relu_out, out
        relu_out, out = self.layer3_1(out)
        # with torch.no_grad():
        #     net['layer20'], net['layer21'] = relu_out, out
        relu_out, out = self.layer3_2(out)
        # with torch.no_grad():
        #     net['layer22'], net['layer23'] = relu_out, out
        relu_out, out = self.layer3_3(out)
        # with torch.no_grad():
        #     net['layer24'], net['layer25'] = relu_out, out

        out = F.avg_pool2d(out, 8)
        out = to_1d(out)
        with torch.no_grad():
            net['embeddings'] = out
        out = self.linear(out)
        net['logits'] = out

        return net

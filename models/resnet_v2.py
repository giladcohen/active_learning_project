import torch
import torch.nn as nn
import torch.nn.functional as F
from active_learning_project.utils import to_1d, activation_batch_ratio, activation_ratio_avg, activation_L1_ratio
from collections import OrderedDict

class res_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, use_bn=True):
        super(res_basic, self).__init__()
        self.use_bn = use_bn
        self.use_bias = False  # not self.use_bn

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=self.use_bias)
        if self. use_bn:
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
    def __init__(self, num_classes=10, use_bn=True):
        super(ResNet18, self).__init__()
        self.depth = 17
        self.dropout_rate = 0.0
        self.use_bn = use_bn
        self.use_bias = False  # not self.use_bn

        self.nStages = [64, 64, 128, 256, 512]

        self.conv1 = nn.Conv2d(3, self.nStages[0], kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        if self.use_bn:
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
            'act1': ['conv1.weight', 'bn1.weight', 'bn1.bias'],
            'act2': ['layer1_0.conv1.weight', 'layer1_0.bn1.weight', 'layer1_0.bn1.bias'],
            'act3': ['layer1_0.conv2.weight', 'layer1_0.bn2.weight', 'layer1_0.bn2.bias'],
            'act4': ['layer1_1.conv1.weight', 'layer1_1.bn1.weight', 'layer1_1.bn1.bias'],
            'act5': ['layer1_1.conv2.weight', 'layer1_1.bn2.weight', 'layer1_1.bn2.bias'],
            'act6': ['layer2_0.conv1.weight', 'layer2_0.bn1.weight', 'layer2_0.bn1.bias'],
            'act7': ['layer2_0.conv2.weight', 'layer2_0.bn2.weight', 'layer2_0.bn2.bias',
                         'layer2_0.shortcut.0.weight', 'layer2_0.shortcut.1.weight', 'layer2_0.shortcut.1.bias'],
            'act8': ['layer2_1.conv1.weight', 'layer2_1.bn1.weight', 'layer2_1.bn1.bias'],
            'act9': ['layer2_1.conv2.weight', 'layer2_1.bn2.weight', 'layer2_1.bn2.bias'],
            'act10': ['layer3_0.conv1.weight', 'layer3_0.bn1.weight', 'layer3_0.bn1.bias'],
            'act11': ['layer3_0.conv2.weight', 'layer3_0.bn2.weight', 'layer3_0.bn2.bias',
                          'layer3_0.shortcut.0.weight', 'layer3_0.shortcut.1.weight', 'layer3_0.shortcut.1.bias'],
            'act12': ['layer3_1.conv1.weight', 'layer3_1.bn1.weight', 'layer3_1.bn1.bias'],
            'act13': ['layer3_1.conv2.weight', 'layer3_1.bn2.weight', 'layer3_1.bn2.bias'],
            'act14': ['layer4_0.conv1.weight', 'layer4_0.bn1.weight', 'layer4_0.bn1.bias'],
            'act15': ['layer4_0.conv2.weight', 'layer4_0.bn2.weight', 'layer4_0.bn2.bias',
                          'layer4_0.shortcut.0.weight', 'layer4_0.shortcut.1.weight', 'layer4_0.shortcut.1.bias'],
            'act16': ['layer4_1.conv1.weight', 'layer4_1.bn1.weight', 'layer4_1.bn1.bias'],
            'act17': ['layer4_1.conv2.weight', 'layer4_1.bn2.weight', 'layer4_1.bn2.bias']
        }

    def forward(self, x):
        net = OrderedDict()
        net['images'] = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = F.relu(out)
        net['L1_act1'] = activation_L1_ratio(out)
        with torch.no_grad():
            net['num_act1'] = activation_ratio_avg(out)

        relu_out, out = self.layer1_0(out)
        net['L1_act2'] = activation_L1_ratio(relu_out)
        net['L1_act3'] = activation_L1_ratio(out)
        with torch.no_grad():
            net['num_act2'] = activation_ratio_avg(relu_out)
            net['num_act3'] = activation_ratio_avg(out)

        relu_out, out = self.layer1_1(out)
        net['L1_act4'] = activation_L1_ratio(relu_out)
        net['L1_act5'] = activation_L1_ratio(out)
        with torch.no_grad():
            net['num_act4'] = activation_ratio_avg(relu_out)
            net['num_act5'] = activation_ratio_avg(out)

        relu_out, out = self.layer2_0(out)
        net['L1_act6'] = activation_L1_ratio(relu_out)
        net['L1_act7'] = activation_L1_ratio(out)
        with torch.no_grad():
            net['num_act6'] = activation_ratio_avg(relu_out)
            net['num_act7'] = activation_ratio_avg(out)

        relu_out, out = self.layer2_1(out)
        net['L1_act8'] = activation_L1_ratio(relu_out)
        net['L1_act9'] = activation_L1_ratio(out)
        with torch.no_grad():
            net['num_act8'] = activation_ratio_avg(relu_out)
            net['num_act9'] = activation_ratio_avg(out)

        relu_out, out = self.layer3_0(out)
        net['L1_act10'] = activation_L1_ratio(relu_out)
        net['L1_act11'] = activation_L1_ratio(out)
        with torch.no_grad():
            net['num_act10'] = activation_ratio_avg(relu_out)
            net['num_act11'] = activation_ratio_avg(out)

        relu_out, out = self.layer3_1(out)
        net['L1_act12'] = activation_L1_ratio(relu_out)
        net['L1_act13'] = activation_L1_ratio(out)
        with torch.no_grad():
            net['num_act12'] = activation_ratio_avg(relu_out)
            net['num_act13'] = activation_ratio_avg(out)

        relu_out, out = self.layer4_0(out)
        net['L1_act14'] = activation_L1_ratio(relu_out)
        net['L1_act15'] = activation_L1_ratio(out)
        with torch.no_grad():
            net['num_act14'] = activation_ratio_avg(relu_out)
            net['num_act15'] = activation_ratio_avg(out)

        relu_out, out = self.layer4_1(out)
        net['L1_act16'] = activation_L1_ratio(relu_out)
        net['L1_act17'] = activation_L1_ratio(out)
        with torch.no_grad():
            net['num_act16'] = activation_ratio_avg(relu_out)
            net['num_act17'] = activation_ratio_avg(out)

        out = F.avg_pool2d(out, 4)
        out = to_1d(out)
        with torch.no_grad():
            net['embeddings'] = out
        out = self.linear(out)
        net['logits'] = out

        return net

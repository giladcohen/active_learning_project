import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def to_1d(x):
    return x.view(x.size(0), -1)

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

    def forward(self, x):
        net = {}

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = F.relu(out)
        net['relu1'] = to_1d(out)

        relu_out, out = self.layer1_0(out)
        net['relu2'] = relu_out
        net['relu3'] = out
        relu_out, out = self.layer1_1(out)
        net['relu4'] = relu_out
        net['relu5'] = out

        relu_out, out = self.layer2_0(out)
        net['relu6'] = relu_out
        net['relu7'] = out
        relu_out, out = self.layer2_1(out)
        net['relu8'] = relu_out
        net['relu9'] = out

        relu_out, out = self.layer3_0(out)
        net['relu10'] = relu_out
        net['relu11'] = out
        relu_out, out = self.layer3_1(out)
        net['relu12'] = relu_out
        net['relu13'] = out

        relu_out, out = self.layer4_0(out)
        net['relu14'] = relu_out
        net['relu15'] = out
        relu_out, out = self.layer4_1(out)
        net['relu16'] = relu_out
        net['relu17'] = out

        out = F.avg_pool2d(out, 4)
        out = to_1d(out)
        net['embeddings'] = out
        out = self.linear(out)
        net['logits'] = out

        return net


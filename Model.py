'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchinfo import summary
import torch.nn.functional as F


# recking result

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ASPP_module, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.dilation = dilation

        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)


class GlobalLocal(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GlobalLocal, self, ).__init__()
        dilations = [1,3,6,9,11,13,14,15]
        # local signals
        # self.out = out_ch
        self.channel_size = out_ch
        self.kernel_size = 2
        self.strides = 2
        self.local_signal = self.cnn = nn.Sequential(
            # batch_size * 1 * 64 * 64
            nn.Conv2d(in_ch, self.channel_size, self.kernel_size, stride=self.strides),
           # ResBlockSqEx(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 32 * 32
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            ResBlockSqEx(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 16 * 16
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            ResBlockSqEx(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 4 * 4
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            ResBlockSqEx(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 2 * 2
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
           # ResBlockSqEx(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 1 * 1
        )
        self.plane_asp = 32
        # global signals
        self.dilated_1 = ASPP_module(self.plane_asp, self.plane_asp, dilation=dilations[0])
      #  self.rb1 = ResBlockSqEx(self.plane_asp)
        self.dilated_2 = ASPP_module(self.plane_asp, self.plane_asp, dilation=dilations[1])
        self.rb2 = ResBlockSqEx(self.plane_asp)
        self.dilated_3 = ASPP_module(self.plane_asp, self.plane_asp, dilation=dilations[2])
        self.rb3 = ResBlockSqEx(self.plane_asp)
        self.dilated_4 = ASPP_module(self.plane_asp, self.plane_asp, dilation=dilations[3])
        self.rb4 = ResBlockSqEx(self.plane_asp)
        self.dilated_5 = ASPP_module(self.plane_asp, self.plane_asp, dilation=dilations[4])
        self.rb5 = ResBlockSqEx(self.plane_asp)
        self.dilated_6 = ASPP_module(self.plane_asp, self.plane_asp, dilation=dilations[5])
       # self.rb6 = ResBlockSqEx(self.plane_asp)
        self.dilated_7 = ASPP_module(self.plane_asp, self.plane_asp, dilation=dilations[6])
        self.dilated_8 = ASPP_module(self.plane_asp, self.plane_asp, dilation=dilations[7])



        self.reshape = nn.Conv2d(288, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.local_signal(x)
        low_level = x
        x1 = self.dilated_1(x)
#        x1 = self.rb1(x1)
        x2 = self.dilated_2(x)
        x2 = self.rb2(x2)
        x3 = self.dilated_3(x)
        x3 = self.rb3(x3)
        x4 = self.dilated_4(x)
        x4 = self.rb4(x4)
        x5 = self.dilated_5(x)
        x5 = self.rb5(x5)
        x6 = self.dilated_6(x)
        x7 = self.dilated_7(x)
        x8 = self.dilated_8(x)
      #  x6 = self.rb6(x6)
        x = torch.concat((x1, x2, x3, x4, x5, x6,x7,x8), dim=1)
        x = torch.concat((x, low_level), dim=1)
        x = self.reshape(x)
        # x = x.view((-1, self.out))
        #       x = self.output(x)
        return x


class SqEx(nn.Module):

    def __init__(self, n_features, reduction=2):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


# Residual block using Squeeze and Excitation

class ResBlockSqEx(nn.Module):

    def __init__(self, n_features):
        super(ResBlockSqEx, self).__init__()

        # convolutions

        self.norm1 = nn.BatchNorm2d(n_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm2 = nn.BatchNorm2d(n_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        # squeeze and excitation

        self.sqex = SqEx(n_features)

    def forward(self, x):
        # convolutions

        y = self.conv1(self.relu1(self.norm1(x)))
        y = self.conv2(self.relu2(self.norm2(y)))

        # squeeze and excitation

        y = self.sqex(y)

        # add residuals

        y = torch.add(x, y)

        return y


x = torch.randn(2, 1, 300, 300)
model = GlobalLocal(1, 32)
# model = SELayer(32)
y = model(x)
summary(model)
print(y.shape)
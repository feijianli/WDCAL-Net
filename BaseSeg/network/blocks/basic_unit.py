"""Basic Module for Semantic Segmentation"""
import torch.nn as nn
import torch
import math
from functools import reduce

__all__ = ['conv3x3x3', 'conv1x1x1', '_ConvINReLU3D', '_ConvIN3D']


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _ConvINReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, p=0.2):
        super(_ConvINReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.drop = nn.Dropout3d(p=p, inplace=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.relu(x)

        return x


class _ConvIN3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(_ConvIN3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):

        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # print(y.shape)
        # Two different branches of ECA module
        # a = self.conv(y.squeeze(-1).transpose(-1, -2))
        # print(y.squeeze(-1).transpose(-1,-2).shape)
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class _DWConvIN3D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,dilation=1):
        super(_DWConvIN3D, self).__init__()
        self.conv = nn.Conv3d(in_channels,out_channels,kernel_size,stride,padding,bias=False,groups=in_channels)
        self.norm = nn.InstanceNorm3d(out_channels)
    def forward(self,x):
        x=self.conv(x)
        x=self.norm(x)
        return x

class SplitAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, b=1, gamma=2):
        super(SplitAttention, self).__init__()
        kernel_size = int(abs((math.log(in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.conv = nn.Conv1d(1, 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.out_channels = out_channels

    def forward(self, x1, x2, x3, x4):
        batch_size = x1.size(0)
        output = [x1, x2, x3, x4]
        # the part of fusion
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        x = x1 + x2 + x3 + x4  # 逐元素相加生成 混合特征
        x = self.global_pool(x)
        # x = self.fc1(x)  # 降维
        # x= self.fc2(x)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        x = self.conv(x.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        # Multi-scale information fusion
        # x = self.sigmoid(x)

        # return x * y.expand_as(x)
        x = x.reshape(batch_size, 4, self.out_channels, -1)  # 调整形状，变为 四个全连接层的值
        x = self.softmax(x)  # 使得两个全连接层对应位置进行softmax
        # the part of selection
        x = list(x.chunk(4, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        x = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1, 1), x))  # 将所有分块  调整形状，即扩展两维
        x = list(map(lambda x, y: x * y, output, x))  # 权重与对应  不同卷积核输出的U 逐元素相乘
        x = reduce(lambda x, y: x + y, x)  # 两个加权后的特征 逐元素相加
        return x


class SplitAttention_DE(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,b=1, gamma=2):
        super(SplitAttention_DE,self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # 自适应pool到指定维度    这里指定为1，实现 GAP
        # self.fc1 = nn.Sequential(nn.Conv3d(in_channels, 32, 1, stride=1,bias=False),
        #                          nn.BatchNorm3d(32),
        #                          nn.ReLU(inplace=True))  # 降维
        # self.fc2 = nn.Conv3d(32, out_channels * 2, 1, 1, bias=False)  # 升维
        kernel_size = int(abs((math.log(in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv = nn.Conv1d(1, 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.out_channels=out_channels

    def forward(self, x1,x2):
        batch_size = x1.size(0)
        output=[x1,x2]
        # the part of fusion
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        x =x1+x2 # 逐元素相加生成 混合特征
        x = self.global_pool(x)
        # x = self.fc1(x)  # 降维
        # x= self.fc2(x)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        x = self.conv(x.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        x = x.reshape(batch_size, 2, self.out_channels,-1)  # 调整形状，变为 四个全连接层的值
        x = self.softmax(x)  # 使得两个全连接层对应位置进行softmax
        # the part of selection
        x = list(x.chunk(2, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        x= list(map(lambda x: x.reshape(batch_size, self.out_channels,1, 1, 1), x))  # 将所有分块  调整形状，即扩展两维
        x = list(map(lambda x, y: x * y, output, x))  # 权重与对应  不同卷积核输出的U 逐元素相乘
        x = reduce(lambda x, y: x + y,x)  # 两个加权后的特征 逐元素相加
        return x

if __name__ == '__main__':
    from torch.autograd import Variable

    img = Variable(torch.zeros(2, 1, 32, 32, 32))
    net = eca_layer()
    out = net(img)
    print(out.size())
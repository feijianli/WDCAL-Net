import torch
import torch.nn as nn

from .basic_unit import _ConvIN3D, _ConvINReLU3D,SplitAttention,_DWConvIN3D,SplitAttention_DE
from .basic_unit import _ConvIN3D, _ConvINReLU3D,eca_layer
from .Attention_block import  GlobalBlock


class ResTwoLayerConvBlock(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1, is_dynamic_empty_cache=False):
        """residual block, including two layer convolution, instance normalization, drop out and ReLU"""
        super(ResTwoLayerConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.residual_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1))
        self.shortcut_unit = _ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output = self.relu(output)

        return output

class ResECATwoLayerConvBlock(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1, is_dynamic_empty_cache=False):
        """residual block, including two layer convolution, instance normalization, drop out and ReLU"""
        super(ResECATwoLayerConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.residual_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1),
             eca_layer(out_channel,k_size=3))
        self.shortcut_unit = _ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output = self.relu(output)

        return output


class ResFourLayerConvBlock(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1, is_dynamic_empty_cache=False):
        """residual block, including four layer convolution, instance normalization, drop out and ReLU"""
        super(ResFourLayerConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.residual_unit_1 = nn.Sequential(
            _ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1))
        self.residual_unit_2 = nn.Sequential(
            _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=p),
            _ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1))
        self.shortcut_unit_1 = _ConvIN3D(in_channel, inter_channel, 1, stride=stride, padding=0)
        self.shortcut_unit_2 = nn.Sequential()
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        output_1 = self.residual_unit_1(x)
        output_1 += self.shortcut_unit_1(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output_1 = self.relu_1(output_1)
        output_2 = self.residual_unit_2(output_1)
        output_2 += self.shortcut_unit_2(output_1)
        if self.is_dynamic_empty_cache:
            del output_1
            torch.cuda.empty_cache()

        output_2 = self.relu_2(output_2)

        return output_2


class ResBaseConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, p=0.2, stride=1, is_identify=True, is_dynamic_empty_cache=False):
        """residual base block, including two layer convolution, instance normalization, drop out and leaky ReLU"""
        super(ResBaseConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.residual_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, out_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(out_channel, out_channel, 3, stride=1, padding=1)
           )
        self.shortcut_unit = nn.Sequential() if stride == 1 and in_channel == out_channel and is_identify else \
            _ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output = self.relu(output)

        return output

    #
    # def __init__(self, in_channel, out_channel, p=0.2, stride=1, is_identify=True, is_dynamic_empty_cache=False):
    #     """residual base block, including two layer convolution, instance normalization, drop out and leaky ReLU"""
    #     super(ResBaseConvBlock, self).__init__()
    #     self.is_dynamic_empty_cache = is_dynamic_empty_cache
    #     self.residual_unit = nn.Sequential(
    #         _ConvINReLU3D(in_channel, out_channel, 3, stride=stride, padding=1, p=p),
    #         _ConvIN3D(out_channel, out_channel, 3, stride=1, padding=1),
    #         GlobalBlock(out_channel))
    #     self.shortcut_unit = nn.Sequential() if stride == 1 and in_channel == out_channel and is_identify else \
    #         _ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
    #     self.relu = nn.ReLU(inplace=True)
    #
    # def forward(self, x):
    #     output = self.residual_unit(x)
    #     output += self.shortcut_unit(x)
    #     if self.is_dynamic_empty_cache:
    #         del x
    #         torch.cuda.empty_cache()
    #
    #     output = self.relu(output)
    #
    #     return output

    # def __init__(self, in_channel,out_channel, p=0.2, stride=1, is_dynamic_empty_cache=False):
    #
    #     super(ResBaseConvBlock, self).__init__()
    #     self.is_dynamic_empty_cache = is_dynamic_empty_cache
    #     self.branch1 = nn.Sequential(
    #         _ConvINReLU3D(in_channel,out_channel,1,stride=1,padding=0,p=p),
    #         _DWConvIN3D(out_channel,out_channel , kernel_size=3, stride=stride, padding=1),
    #         _ConvINReLU3D(out_channel, in_channel, 1, stride=1, padding=0, p=0),
    #
    #     )
    #     self.branch2= nn.Sequential(
    #         _ConvINReLU3D(in_channel, out_channel, 1, stride=1, padding=0, p=p),
    #         _DWConvIN3D(out_channel, out_channel, kernel_size=3, stride=stride, padding=1,dilation=1),
    #         _ConvINReLU3D(out_channel, out_channel, 1, stride=1, padding=0, p=0),
    #         _DWConvIN3D(out_channel, out_channel, kernel_size=3, stride=1, padding=1,dilation=1),
    #         _ConvINReLU3D(out_channel, in_channel, 1, stride=1, padding=0, p=0),
    #     )
    #     self.branch3= nn.Sequential(
    #         _ConvINReLU3D(in_channel, out_channel, 1, stride=1, padding=0, p=p),
    #         _DWConvIN3D(out_channel, out_channel, kernel_size=3, stride=stride, padding=1,dilation=2),
    #
    #         _ConvINReLU3D(out_channel, out_channel, 1, stride=1, padding=0, p=0),
    #         _DWConvIN3D(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
    #         _ConvINReLU3D(out_channel, in_channel, 1, stride=1, padding=0, p=0),
    #     )
    #     self.branch4 = nn.Sequential(
    #         _ConvINReLU3D(in_channel, out_channel, 1, stride=1, padding=0, p=p),
    #         _DWConvIN3D(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, dilation=2),
    #         _ConvINReLU3D(out_channel, out_channel, 1, stride=1, padding=0, p=0),
    #         _DWConvIN3D(out_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=2),
    #         _ConvINReLU3D(out_channel, in_channel, 1, stride=1, padding=0, p=0),
    #     )
    #     self.attention=SplitAttention(in_channel,in_channel,kernel_size=1,stride=1,padding=0)
    #     self.ConvIN3D=_ConvIN3D(in_channel,out_channel,kernel_size=1,stride=1,padding=0)
    #     self.shortcut = _ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
    #     self.relu= nn.ReLU(inplace=True)
    #
    # def forward(self, x):
    #     output_1=self.branch1(x)
    #     output_2=self.branch2(x)
    #     output_3=self.branch3(x)
    #     output_4=self.branch4(x)
    #     output=self.attention(output_1,output_2,output_3,output_4)
    #     output = self.ConvIN3D(output)
    #     output += self.shortcut(x)
    #     if self.is_dynamic_empty_cache:
    #         del x
    #         torch.cuda.empty_cache()
    #
    #     output = self.relu(output)
    #
    #     return output

# class UnetConv3(nn.Module):
#         def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 1), padding_size=(1, 1, 0),
#                      init_stride=(1, 1, 1)):
#             super(UnetConv3, self).__init__()
#
#             if is_batchnorm:
#                 self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
#                                            nn.BatchNorm3d(out_size),
#                                            nn.ReLU(inplace=True), )
#                 self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
#                                            nn.BatchNorm3d(out_size),
#                                            nn.ReLU(inplace=True), )
#             else:
#                 self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
#                                            nn.ReLU(inplace=True), )
#                 self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
#                                            nn.ReLU(inplace=True), )
#
#             # initialise the blocks
#             for m in self.children():
#                 init_weights(m, init_type='kaiming')
#
#         def forward(self, inputs):
#             outputs = self.conv1(inputs)
#             outputs = self.conv2(outputs)
#             return outputs

class AnisotropicConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, p=0.2, stride=1, is_identify=True, is_dynamic_empty_cache=False):
        """Anisotropic convolution block, including two layer convolution,
         instance normalization, drop out and ReLU"""
        super(AnisotropicConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.residual_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, out_channel, kernel_size=(3, 3, 1), stride=stride, padding=(1, 1, 0), p=p),
            _ConvIN3D(out_channel, out_channel, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)))
        self.shortcut_unit = nn.Sequential() if stride == 1 and in_channel == out_channel and is_identify else \
            _ConvIN3D(in_channel, out_channel, kernel_size=1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output = self.relu(output)

        return output

    # def __init__(self, in_channel, out_channel, p=0.2, stride=1, is_identify=True, is_dynamic_empty_cache=False):
    #     """Anisotropic convolution block, including two layer convolution,
    #      instance normalization, drop out and ReLU"""
    #     super(AnisotropicConvBlock, self).__init__()
    #     self.is_dynamic_empty_cache = is_dynamic_empty_cache
    #     self.residual_unit = nn.Sequential(
    #         _ConvINReLU3D(in_channel, out_channel, kernel_size=(3, 3, 1), stride=stride, padding=(1, 1, 0), p=p),
    #         _ConvIN3D(out_channel, out_channel, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)),
    #         GlobalBlock(out_channel))
    #     self.shortcut_unit = nn.Sequential() if stride == 1 and in_channel == out_channel and is_identify else \
    #         _ConvIN3D(in_channel, out_channel, kernel_size=1, stride=stride, padding=0)
    #     self.relu = nn.ReLU(inplace=True)
    #
    # def forward(self, x):
    #     output = self.residual_unit(x)
    #     output += self.shortcut_unit(x)
    #     if self.is_dynamic_empty_cache:
    #         del x
    #         torch.cuda.empty_cache()
    #
    #     output = self.relu(output)
    #
    #     return output

    # def __init__(self, in_channel,out_channel, p=0.2, stride=1, is_dynamic_empty_cache=False):
    #
    #     super(AnisotropicConvBlock, self).__init__()
    #     self.is_dynamic_empty_cache = is_dynamic_empty_cache
    #     self.branch1 = nn.Sequential(
    #         _ConvINReLU3D(in_channel,out_channel,1,stride=1,padding=0,p=p),
    #         _DWConvIN3D(out_channel,out_channel , kernel_size=3, stride=stride, padding=1),
    #         _ConvINReLU3D(out_channel, in_channel, 1, stride=1, padding=0, p=0),
    #
    #     )
    #     self.branch2= nn.Sequential(
    #         _ConvINReLU3D(in_channel, out_channel, 1, stride=1, padding=0, p=p),
    #         _DWConvIN3D(out_channel, out_channel, kernel_size=3, stride=stride, padding=1,dilation=1),
    #         _ConvINReLU3D(out_channel, out_channel, 1, stride=1, padding=0, p=0),
    #         _DWConvIN3D(out_channel, out_channel, kernel_size=3, stride=1, padding=1,dilation=1),
    #         _ConvINReLU3D(out_channel, in_channel, 1, stride=1, padding=0, p=0),
    #     )
    #     self.attention = SplitAttention_DE(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
    #     self.ConvIN3D = _ConvIN3D(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
    #     self.shortcut = nn.Sequential()
    #     self.relu = nn.ReLU(inplace=True)
    #
    # def forward(self, x):
    #     output_1 = self.branch1(x)
    #     output_2 = self.branch2(x)
    #     output = self.attention(output_1, output_2)
    #     output = self.ConvIN3D(output)
    #     output += self.shortcut(x)
    #     if self.is_dynamic_empty_cache:
    #         del x
    #         torch.cuda.empty_cache()
    #
    #     output = self.relu(output)
    #
    #     return output




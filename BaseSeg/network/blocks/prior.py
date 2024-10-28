# from mmcv.cnn.bricks.norm import build_norm_layer
import torch
import torch.nn.functional as F

from mmcv.cnn import ConvModule, Scale
from torch import nn
from torch.nn.modules.utils import _pair
import numpy as np

# from mmseg.core import add_prefix
# from mmseg.ops import resize
# from ..builder import HEADS, build_loss
# from ..utils import SelfAttentionBlock as _SelfAttentionBlock
# from .decode_head import BaseDecodeHead

torch.set_default_dtype(torch.float32)


class PriorConvBlock(nn.Module):
    """Context Prior for Scene Segmentation.

    This head is the implementation of `CPNet
    <https://arxiv.org/abs/2004.01547>`_.
    """

    def __init__(self,
                 in_channels,   #4
                 prior_channels,  #5

                 # prior_size,
                 # am_kernel_size,
                 groups=1,
                 loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0),
                 **kwargs):
        super(PriorConvBlock, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.prior_channels = prior_channels

        # self.prior_size = _pair(prior_size)
        # self.am_kernel_size = am_kernel_size

        # self.aggregation = AggregationModule(self.in_channels, prior_channels,
        #                                      am_kernel_size, self.conv_cfg,
        #                                      self.norm_cfg)

        self.prior_conv = ConvModule(
            self.in_channels,
            # np.prod(self.prior_size),
            self.prior_channels,
            (1,1,1),
            padding=0,
            stride=(1,1,1),

            conv_cfg=dict(type='Conv3d'),
            norm_cfg= dict(type='IN3d'),
            act_cfg=dict(type='ReLU'),
            inplace=True)

        self.intra_conv = ConvModule(
            self.prior_channels,
            self.in_channels,
            (1,1,1),
            padding=0,
            stride=(1,1,1),
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='IN3d'),
            act_cfg=dict(type='ReLU'),
            inplace=True)

        self.inter_conv = ConvModule(
            self.prior_channels,
            self.in_channels,
            (1,1,1),
            padding=0,
            stride=(1,1,1),
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='IN3d'),
            act_cfg=dict(type='ReLU'),
            inplace=True)



        # self.loss_prior_decode = build_loss(loss_prior_decode)

    def forward(self, x):
        """Forward function."""
        # x = self._transform_inputs(x)
        batch_size, channels, height, width,depth = x.size()  #1*4*192*192*192
        x_size = x.size()[2:]
        # print(x.shape)

        assert x_size[0] == height and x_size[1] == width

        value = F.interpolate(
            x.float(), size= channels , mode='trilinear', align_corners=True) #1*4*4*4*4

        # value = self.aggregation(x)

        # context_prior_map = self.prior_conv(x)   #1*5*96*96*96
        # context_prior_map = context_prior_map.view(batch_size,
        #                                            np.prod(self.prior_size),
        #                                            -1)
        # value = self.prior_conv(value)  #1*4*4*4*4

        # print(value.shape)
        value = value.view(batch_size, self.prior_channels,-1)  # 1*4*64
        # print(value.shape)


        value=value.unsqueeze(1)  #1*1*4*64
        value= F.interpolate(
            value.float(), size= channels, mode='bilinear', align_corners=True)  # 2*1*4*4
        context_prior_map= value.squeeze(1)  # 2*4*4
        # # print(context_prior_map)
        # for i in range(context_prior_map.shape[0]):
        #     for j in range(context_prior_map.shape[1]):
        #         for k in range(context_prior_map.shape[2]):
        #             if k==j:
        #                 context_prior_map[i][j][k] = context_prior_map[i][j][k] + 0.0001
        #             else: context_prior_map[i][j][k] = 0
        # # print(context_prior_map)
        # context_prior_mapf = context_prior_map.view(batch_size, 1, -1)  # 1*1*16
        #
        # context_prior_mapf = torch.sum(context_prior_mapf, dim=2)
        # # print(context_prior_mapf)
        # context_prior_mapf = context_prior_mapf.unsqueeze(1)
        #
        # context_prior_map = torch.div(context_prior_map, context_prior_mapf)


        # context_prior_map = context_prior_map.view(batch_size,1, -1)  #2*1*16
        # context_prior_map = F.softmax(context_prior_map, dim=2)
        # context_prior_map = context_prior_map.view(batch_size,self.prior_channels, -1)  # 1*4*4
        #
        context_prior_map = context_prior_map.permute(0, 2, 1)
        context_prior_map = torch.sigmoid(context_prior_map)
        # inter_context_prior_map = 1 - context_prior_map
        # inter_context_prior_map = inter_context_prior_map.view(batch_size, self.prior_channels, -1)
        # x_1 = self.prior_conv(x)  #1*4*192*192*192
        # print(x_1.shape)

        x_1 = x.view(batch_size, self.prior_channels, -1)  #1*4*(192*192*192)
        # x_1 = torch.tensor(x_1)
        # print(x_1.shape)
        # value = x.view(batch_size, self.prior_channels, -1)
        # value = value.permute(0, 2, 1)


        intra_context = torch.bmm(context_prior_map.float(), x_1.float())   #1*4*(192*192*192)
        # intra_context = intra_context.div(self.prior_channels)
        # intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(batch_size, self.prior_channels,
                                           x_size[0],
                                           x_size[1],
                                           x_size[2])
        # intra_context = self.intra_conv(intra_context)  #1*4*192*192*192

        # intra_context = torch.bmm(context_prior_map, value)
        # intra_context = intra_context.div(np.prod(self.prior_size))
        # intra_context = intra_context.permute(0, 2, 1).contiguous()
        # intra_context = intra_context.view(batch_size, self.prior_channels,
        #                                    self.prior_size[0],
        #                                    self.prior_size[1])
        # intra_context = self.intra_conv(intra_context)

        # inter_context = torch.bmm(inter_context_prior_map, x_1)
        # inter_context = inter_context.div(self.prior_channels)
        # # inter_context = inter_context.permute(0, 2, 1).contiguous()
        # inter_context = inter_context.view(batch_size, self.prior_channels,
        #                                    x_size[0],
        #                                    x_size[1],
        #                                    x_size[2])
        # inter_context = self.inter_conv(inter_context)   #1*4*192*192*192
        #
        # outs = torch.cat([intra_context, inter_context], dim=1)
        # output = self.bottleneck(outs)
        output = intra_context + x

        return [output, context_prior_map]

class C1_PriorConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            """residual block, including two layer convolution, instance normalization, drop out and ReLU"""
            super(C1_PriorConvBlock, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            # self.conv = ConvModule(
            #             self.in_channels,
            #             self.out_channels,
            #             (1,1,1),
            #             padding=0,
            #             stride=(1,1,1),
            #             conv_cfg=dict(type='Conv3d'),
            #             norm_cfg=dict(type='IN3d'),
            #             act_cfg=None)
            self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
            self.SoftMax = nn.Softmax(dim=1)

        def forward(self, x):
            y = self.avg_pool(x)  # 2*4*1*1*1
            y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
            # y = self.conv(y)
            map = self.SoftMax(y)  # 2*4*1*1*1
            mapx  = map.squeeze(-1).squeeze(-1).squeeze(-1)
            output = x + x * map.expand_as(x)  # 2*4*192*192*192
            return [output, mapx]

class Bottleneck(nn.Module):
    def __init__(self, channels, is_dynamic_empty_cache=False):

        super(Bottleneck, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.bottleneck = ConvModule(
            channels * 3,
            channels* 4,
            (3, 3, 3),
            padding=(1, 1, 1),
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='IN3d'),
            act_cfg=dict(type='ReLU'),
            inplace=True)
        self.final = nn.Conv3d(channels* 4, channels, kernel_size=1, bias= False)
        # self.Conv_value = nn.Sequential(
        #     nn.Conv3d(channels * 3, channels * 4, 1),
        #     # nn.LayerNorm([self.out_channels, 1, 1, 1]),
        #     nn.InstanceNorm3d(channels * 4),
        #     # nn.BatchNorm3d(self.out_channels),
        #     nn.ReLU(),
        #     nn.Conv3d(channels * 4, channels, 1),
        # )

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.final(x)
        return x




def ideal_affinity_matrix( label):
    labels = F.interpolate(
        label.float(), size=label.shape[2:], mode="nearest")  #1*4*192*192*192

    labels = labels.squeeze_().long()
    # labels[labels == 255] = num_classes
    # one_hot_labels = F.one_hot(labels, num_classes + 1)  #1*4*192*192*192
    labels = labels.view(
        labels.size(0),labels.size(1), -1).float()   #1*4*(192*192*192)
    ideal_affinity_matrix = torch.bmm(labels.div(100),labels.permute(0, 2, 1))  #1*4*4

    # ideal_affinity_matrix = ideal_affinity_matrix.permute(0, 2, 1)
    ideal_affinity_matrix = torch.sigmoid(ideal_affinity_matrix)



    # ideal_affinity_matrixN= ideal_affinity_matrix.view(labels.shape[0], labels.shape[1], -1)
    # ideal_affinity = F.softmax(ideal_affinity_matrix, dim=2)

    # #xiugai
    # ideal_affinity_matrixf = ideal_affinity_matrix.view(labels.shape[0], 1, -1)  # 1*1*16
    # ideal_affinity_matrixf = torch.sum(ideal_affinity_matrixf, dim=2)
    # ideal_affinity_matrixf = ideal_affinity_matrixf.unsqueeze(1)
    # # ideal_affinity_matrix = ideal_affinity_matrix.view(labels.shape[0], labels.shape[1], -1)
    # ideal_affinity_matrix = torch.div(ideal_affinity_matrix,ideal_affinity_matrixf)
    # ideal_affinity_matrix= ideal_affinity_matrix.view(labels.shape[0], labels.shape[1], -1)  # 1*4*4


    # print(ideal_affinity_matrix.shape)
    return ideal_affinity_matrix

def C1_ideal_affinity_matrix(label):
    SoftMax = nn.Softmax(dim=1)
    labels = F.interpolate(
        label.float(), size=label.shape[2:], mode="nearest")  # 1*4*192*192*192

    labels = labels.squeeze_().float()
    # labels[labels == 255] = num_classes
    # one_hot_labels = F.one_hot(labels, num_classes + 1)  #1*4*192*192*192
    avg_pool = nn.AdaptiveAvgPool3d(1)
    ideal_affinity_matrix = avg_pool(labels)    #1*4*1*1*1
    ideal_affinity_matrix = SoftMax(ideal_affinity_matrix)
    ideal_affinity_matrix = ideal_affinity_matrix.squeeze(-1).squeeze(-1).squeeze(-1)

    return ideal_affinity_matrix


def prior_losses(p, g):
    """Compute ``seg``, ``prior_map`` loss."""
    # assert len(p) ==len(g)
    # # p = np.array(p)
    # # g = np.array(g)
    # p = p.cpu().detach().numpy()
    # g = g.cpu().detach().numpy()
    #
    # loss = np.sqrt(np.sum(np.square(p-g)) / len(p))
    #
    # return loss


    loss_fn =torch.nn.MSELoss(reduction='mean')
    # loss_fn = torch.nn.L1Loss(reduction='mean')
    loss = loss_fn(p.float(),g.float())
    loss = torch.sqrt(loss)
    # loss.backward()

    return loss

def C1_prior_losses(p, g):
    # kl = F.kl_div(p.softmax(dim=-1).log(), g.softmax(dim=-1), reduction='mean')
    kl = F.kl_div(p.log(), g, reduction='sum')

    loss_fn = torch.nn.MSELoss(reduction='mean')
    # loss_fn = torch.nn.L1Loss(reduction='mean')
    loss = loss_fn(p.float(), g.float())
    loss = torch.sqrt(loss)
    loss = kl + loss
    return loss

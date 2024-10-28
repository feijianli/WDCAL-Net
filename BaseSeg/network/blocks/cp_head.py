from mmcv.cnn.bricks.norm import build_norm_layer
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from torch import nn
from torch.nn.modules.utils import _pair
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# from mmseg.core import add_prefix
# from mmseg.ops import resize
# from ..builder import HEADS, build_loss
# from ..utils import SelfAttentionBlock as _SelfAttentionBlock
# from .decode_head import BaseDecodeHead


class AggregationModule(nn.Module):
    """Aggregation Module"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super(AggregationModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = kernel_size // 2

        # self.reduce_conv = ConvModule(
        #     in_channels,
        #     out_channels,
        #     kernel_size=3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=dict(type='ReLU'))

        self.t1 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=out_channels,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=None,
            act_cfg=None)
        self.t2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size, 1),
            padding=(0, padding, 0),
            groups=out_channels,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=None,
            act_cfg=None)
        self.t3 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1,1, kernel_size),
            padding=(0,0, padding),
            groups=out_channels,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=None,
            act_cfg=None)


        self.p1 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, 1, kernel_size),
            padding=(0, 0, padding),
            groups=out_channels,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=None,
            act_cfg=None)
        self.p2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size, 1),
            padding=(0, padding, 0),
            groups=out_channels,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=None,
            act_cfg=None)
        self.p3 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=out_channels,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=None,
            act_cfg=None)

        # _, self.norm = build_norm_layer(norm_cfg, out_channels)
        self.BN =nn.BatchNorm3d(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward function."""
        x1 = self.t1(x)
        x1 = self.t2(x1)
        x1 = self.t3(x1)

        x2 = self.p1(x)
        x2 = self.p2(x2)
        x2 = self.p3(x2)

        out = self.relu(self.BN(x1 + x2))
        return out



class CPHead(nn.Module):
    """Context Prior for Scene Segmentation.

    This head is the implementation of `CPNet
    <https://arxiv.org/abs/2004.01547>`_.
    """

    def __init__(self,
                 prior_channels,
                 prior_size,
                 am_kernel_size,
                 groups=1,
                 # loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0),
                 **kwargs):
        super(CPHead, self).__init__(**kwargs)
        self.prior_channels = prior_channels
        self.prior_size = _pair(prior_size)
        self.am_kernel_size = am_kernel_size

        self.aggregation = AggregationModule(prior_channels, prior_channels,
                                             am_kernel_size )

        self.prior_conv = ConvModule(
            self.prior_channels,
            np.prod(self.prior_size),
            (1,1,1),
            padding=0,
            stride=(1,1,1),
            groups=groups,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=None)

        self.intra_conv = ConvModule(
            self.prior_channels,
            self.prior_channels,
            (1, 1, 1),
            padding=0,
            stride=(1, 1, 1),
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=dict(type='ReLU'))

        self.inter_conv = ConvModule(
            self.prior_channels,
            self.prior_channels,
            (1, 1, 1),
            padding=0,
            stride=(1, 1, 1),
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=dict(type='ReLU'))

        self.bottleneck = ConvModule(
            self.prior_channels * 3,
            self.prior_channels,
            (3,3,3),
            padding=(1,1,1),
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=dict(type='ReLU'))

        # self.loss_prior_decode = build_loss(loss_prior_decode)
        self.cls_seg = nn.Conv3d(self.prior_channels, self.prior_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """Forward function."""
        # x = self._transform_inputs(inputs)
        batch_size, channels, height, width,Depth = x.size()
        assert self.prior_size[0] == height and self.prior_size[1] == width

        value = self.aggregation(x)

        context_prior_map = self.prior_conv(value)
        context_prior_map = context_prior_map.view(batch_size,
                                                   np.prod(self.prior_size),
                                                   -1)
        context_prior_map_nosig = context_prior_map.permute(0, 2, 1)
        context_prior_map = torch.sigmoid(context_prior_map_nosig)

        inter_context_prior_map = 1 - context_prior_map

        value = value.view(batch_size, self.prior_channels, -1)
        value = value.permute(0, 2, 1)

        intra_context = torch.bmm(context_prior_map, value)
        intra_context = intra_context.div(np.prod(self.prior_size))
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(batch_size, self.prior_channels,
                                           self.prior_size[0],
                                           self.prior_size[1],
                                           self.prior_size[2])
        intra_context = self.intra_conv(intra_context)

        inter_context = torch.bmm(inter_context_prior_map, value)
        inter_context = inter_context.div(np.prod(self.prior_size))
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(batch_size, self.prior_channels,
                                           self.prior_size[0],
                                           self.prior_size[1],
                                           self.prior_size[2])
        inter_context = self.inter_conv(inter_context)

        cp_outs = torch.cat([x, intra_context, inter_context], dim=1)
        output = self.bottleneck(cp_outs)
        output = self.cls_seg(output)

        return [output, context_prior_map_nosig]

    # def forward_test(self, inputs, img_metas, test_cfg):
    #     """Forward function for testing, only ``pam_cam`` is used."""
    #     return self.forward(inputs)[0]

def ideal_affinity_matrix(label):
        # scaled_labels = F.interpolate(
        #     label.float(), size=label.shape[2:], mode="nearest")
        scaled_labels = F.interpolate(
            label.float(), size=12, mode="nearest")
        scaled_labels = scaled_labels.squeeze_().long()
        # scaled_labels[scaled_labels == 255] = self.num_classes
        # one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)
        # one_hot_labels = one_hot_labels.view(
        #     one_hot_labels.size(0), -1, self.num_classes + 1).float()
        scaled_labels = scaled_labels.view(
            scaled_labels.size(0),-1,scaled_labels.size(1)).float()
        ideal_affinity_matrix = torch.bmm(scaled_labels,
                                          scaled_labels.permute(0, 2, 1))
        return ideal_affinity_matrix


def cross_entropy(pred,
                      label,
                      weight=None,
                      class_weight=None,
                      reduction='mean',
                      avg_factor=None,
                      ):
        """The wrapper function for :func:`F.cross_entropy`"""
        # class_weight is a manual rescaling weight given to each class.
        # If given, has to be a Tensor of size C element-wise losses
        loss = F.cross_entropy(
            pred,
            label,
            weight=None,
            reduction='elementwise_mean')
        return loss

# def _expand_onehot_labels(labels, label_weights, label_channels):
#         """Expand onehot labels to match the size of prediction."""
#         bin_labels = labels.new_full((labels.size(0), label_channels), 0)
#         inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
#         if inds.numel() > 0:
#             bin_labels[inds, labels[inds] - 1] = 1
#         if label_weights is None:
#             bin_label_weights = None
#         else:
#             bin_label_weights = label_weights.view(-1, 1).expand(
#                 label_weights.size(0), label_channels)
#         return bin_labels, bin_label_weights

def binary_cross_entropy(pred,
                             label,
                             use_sigmoid=False,
                             weight=None,
                             reduction='mean',
                             avg_factor=None,
                             class_weight=None):
        """Calculate the binary CrossEntropy loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, 1).
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            class_weight (list[float], optional): The weight for each class.

        Returns:
            torch.Tensor: The calculated loss
        """
        # if pred.dim() != label.dim():
        #     label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

        # weighted element-wise losses

        if weight is not None:
            weight = weight.float()
        if use_sigmoid:
            loss = F.binary_cross_entropy_with_logits(
                pred.float(), label.float(), weight=class_weight, reduction='elementwise_mean')
        else:
            loss = F.binary_cross_entropy(
                pred.float(), label.float(), weight=class_weight, reduction='elementwise_mean')


        return loss

# @LOSSES.register_module()
class AffinityLoss(nn.Module):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """

        def __init__(self, reduction='mean', loss_weight=1.0):
            super(AffinityLoss, self).__init__()
            self.reduction = reduction
            self.loss_weight = loss_weight
            self.cls_criterion = binary_cross_entropy

        def forward(self,
                    cls_score,
                    label,
                    weight=None,
                    avg_factor=None,
                    reduction_override=None,
                    **kwargs):
            """Forward function."""
            # assert reduction_override in (None, 'none', 'mean', 'sum')
            cls_score = F.sigmoid(cls_score)
            reduction = (
                reduction_override if reduction_override else self.reduction)

            unary_term = self.cls_criterion(
                cls_score,
                label,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)

            # diagonal_matrix = (1 - torch.eye(label.size(1))).to(label.get_device())
            # vtarget = diagonal_matrix * label

            # recall_part = torch.sum(cls_score * vtarget, dim=2)
            # denominator = torch.sum(vtarget, dim=2)
            # denominator = denominator.masked_fill_(~(denominator > 0), 1)
            # recall_part = recall_part.div_(denominator)
            # recall_label = torch.ones_like(recall_part)
            # recall_loss = self.cls_criterion(
            #     recall_part,
            #     recall_label,
            #     reduction=reduction,
            #     avg_factor=avg_factor,
            #     **kwargs)
            #
            # spec_part = torch.sum((1 - cls_score) * (1 - label), dim=2)
            # denominator = torch.sum(1 - label, dim=2)
            # denominator = denominator.masked_fill_(~(denominator > 0), 1)
            # spec_part = spec_part.div_(denominator)
            # spec_label = torch.ones_like(spec_part)
            # spec_loss = self.cls_criterion(
            #     spec_part,
            #     spec_label,
            #     reduction=reduction,
            #     avg_factor=avg_factor,
            #     **kwargs)
            #
            # precision_part = torch.sum(cls_score * vtarget, dim=2)
            # denominator = torch.sum(cls_score, dim=2)
            # denominator = denominator.masked_fill_(~(denominator > 0), 1)
            # precision_part = precision_part.div_(denominator)
            # precision_label = torch.ones_like(precision_part)
            # precision_loss = self.cls_criterion(
            #     precision_part,
            #     precision_label,
            #     reduction=reduction,
            #     avg_factor=avg_factor,
            #     **kwargs)

            # global_term = recall_loss + spec_loss + precision_loss

            # loss_cls = self.loss_weight * (unary_term + global_term)
            # return loss_cls
            return unary_term


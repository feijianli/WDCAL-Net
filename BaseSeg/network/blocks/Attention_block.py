import torch
import torch.nn as nn
import torch.nn.functional as F

from BaseSeg.network.blocks.basic_unit import _ConvIN3D, _ConvINReLU3D


class GlobalBlock(nn.Module):
    def __init__(self, in_channels, scale = 4,is_dynamic_empty_cache=False):

        super(GlobalBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.in_channels = in_channels
        self.out_channels = self.in_channels//scale

        self.Conv_key = nn.Conv3d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)
        # self.relu = nn.ReLU(inplace=True)

        self.Conv_value = nn.Sequential(
            nn.Conv3d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1, 1]),
            # nn.InstanceNorm3d(self.out_channels),
            # nn.BatchNorm3d(self.out_channels),
            nn.ReLU(),
            nn.Conv3d(self.out_channels, self.in_channels, 1),
        )
        #ghblock
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.score_layer = nn.Sequential(_ConvINReLU3D(in_channels, in_channels, 3, stride=1, padding=1, p=0.2),
                                         nn.Conv3d(self.in_channels, self.in_channels, 1, bias=False))

        self.conv3 = _ConvIN3D(self.in_channels, self.in_channels, 1, stride=1, padding=0)

        self.conv_mask = nn.Conv3d(self.in_channels, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w, d= x.size()

        # key -> [b, 1, H, W, D] -> [b, 1, H*W*D] ->  [b, H*W*D, 1]
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        # key = self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1, 1, 1).contiguous()

        # key = self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1, 1, 1)
        # key = self.conv_1d(key.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        # key = self.SoftMax(key.view(b, -1, 1).contiguous())


        query = x.view(b, c, h*w*d)
        # [b, c, h*w*D] * [b, H*W*D, 1]
        concate_QK = torch.matmul(query, key)
        concate_QK = concate_QK.view(b, c, 1, 1, 1).contiguous()
        value = self.Conv_value(concate_QK)
        x1 = x + value
        x1 = self.relu(x1)

        # # ghblock

        keyh = self.avg_pool(x)  # [b,c,1,1,1]

        keyh = self.conv_1d(keyh.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        # keyh = self.Conv_value(keyh)
        keyh = self.SoftMax(keyh.view(b, -1, 1).contiguous())  # [b,c,1]
        queryh = x.view(b, h * w * d, c)  # [b,H*W*D,c]
        # [b, h*w*D,c] * [b, c, 1]---> [b,h*w*D,1]
        concate_QKh = torch.matmul(queryh, keyh)
        concate_QKh = concate_QKh.view(b, h * w * d, 1, 1, 1).contiguous()  # [b,h*w*d,1,1,1]
        valueh = concate_QKh.view(b, 1, h, w, d).contiguous()  # [b,1,h,w,d]
        x2 = x + valueh


        # out = x1 + x2
        out = self.conv3(x1+x2)
        out = self.relu(x + out)
        # out = F.relu(x + out, inplace=True)
        # out = self.score_layer(out)
        return out






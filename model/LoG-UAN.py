import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Fusion1(nn.Module):
    def __init__(self, ch_in, ch_out, eps=1e-8):
        super(Fusion1, self).__init__()

        # self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        return self.up(x)


class Fusion(nn.Module):
    def __init__(self, indim, dim, eps=1e-8):
        super(Fusion, self).__init__()
        self.pre_conv = Conv(indim, dim, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * self.pre_conv(x)
        x = self.post_conv(x)
        return x


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias,
                                   stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

from EGCA import TransformerBlock1
from ULA import Transformer_block_local


class Decoder(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 dropout=0.1):
        super(Decoder, self).__init__()

        self.b4 = TransformerBlock1(16 * 16, 2048, 32, 8)

        self.p3 = Fusion(2048, 1024)
        self.b3 = TransformerBlock1(32 * 32, 1024, 64, 8)

        self.p2 = Fusion(1024, 512)
        self.b2 = TransformerBlock1(64 * 64, 512, 64, 8)

        self.p1 = Fusion(512, 256)
        self.b1 = TransformerBlock1(128 * 128, 256, 64, 8)

        self.uctb = Transformer_block_local(64, 64, 256, 1, heads=8, patch_size=1, win_size=16)
        self.segmentation_head = nn.Sequential(ConvBNReLU(256, 256),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(256, 6, kernel_size=1))
        self.p0 = Fusion1(256, 64)
        self.segmentation_head1 = nn.Sequential(ConvBNReLU(64, 64),
                                                nn.Dropout2d(p=dropout, inplace=True),
                                                Conv(64, 6, kernel_size=1))

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b4(res4)

        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)
        x = self.b1(x)
        x1 = x.clone()

        x = self.segmentation_head(x)
        x11 = self.p0(x1)
        x11 = self.uctb(x11, x)
        x0 = self.segmentation_head1(x11)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x0 = F.interpolate(x0, size=(h, w), mode='bilinear', align_corners=False)

        return x0,x


class LOGUANet(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 dropout=0.1,
                 num_classes=6,
                 backbone=ResNet50
                 ):
        super().__init__()

        self.backbone = backbone()
        self.decode = Decoder()

    def forward(self, x):
        h, w = x.size()[-2:]
        outs = []
        res0, res1, res2, res3, res4 = self.backbone(x)
        out0,out1 = self.decode(res1, res2, res3, res4, h, w)

        outs.append(out0)
        outs.append(out1)

        return outs

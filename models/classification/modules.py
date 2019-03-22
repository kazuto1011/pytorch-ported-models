#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   08 March 2019

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

_BN_KWARGS = {"eps": 1e-5, "momentum": 0.997}


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, SynchronizedBatchNorm2d):
        nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class _BnReLU(nn.Sequential):
    def __init__(self, in_ch, sync_bn=True):
        super().__init__()
        BatchNorm2d = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d
        self.add_module("bn", BatchNorm2d(in_ch, **_BN_KWARGS))
        self.add_module("relu", nn.ReLU(inplace=True))


class _ConvBnReLU(nn.Sequential):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
        sync_bn=True,
    ):
        super().__init__()
        BatchNorm2d = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", BatchNorm2d(out_ch, **_BN_KWARGS))
        if relu:
            self.add_module("relu", nn.ReLU(inplace=True))


class _UpConvBn(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, sync_bn=True):
        super().__init__()
        BatchNorm2d = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d
        self.add_module(
            "conv",
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        )
        self.add_module("bn", BatchNorm2d(out_ch, **_BN_KWARGS))


class _SeparableConv2d(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, bias):
        super(_SeparableConv2d, self).__init__()
        depthwise_kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": in_ch,
            "bias": bias,
        }
        self.add_module("depthwise", nn.Conv2d(in_ch, in_ch, **depthwise_kwargs))
        self.add_module("pointwise", nn.Conv2d(in_ch, out_ch, 1, 1, 0, 1, 1, bias=bias))


class _SepConvBnReLU(_SeparableConv2d):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
        sync_bn=True,
    ):
        super(_SepConvBnReLU, self).__init__(
            in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
        )
        BatchNorm2d = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d
        self.add_module("bn", BatchNorm2d(out_ch, **_BN_KWARGS))
        if relu:
            self.add_module("relu", nn.ReLU(inplace=True))

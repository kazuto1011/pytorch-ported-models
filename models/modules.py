#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    http://kazuto1011.github.io
# Date:   06 March 2019


import torch.nn as nn

try:
    from encoding.nn import SyncBatchNorm

    _BATCH_NORM = SyncBatchNorm
except:
    _BATCH_NORM = nn.BatchNorm2d

_BN_KWARGS = {"eps": 1e-5, "momentum": 0.997}


class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


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


class _ConvBnReLU(nn.Sequential):
    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, **_BN_KWARGS))
        if relu:
            self.add_module("relu", nn.ReLU(inplace=True))


class _SepConvBnReLU(_SeparableConv2d):
    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_SepConvBnReLU, self).__init__(
            in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
        )
        self.BATCH_NORM = _BATCH_NORM
        self.add_module("bn", _BATCH_NORM(out_ch, **_BN_KWARGS))
        if relu:
            self.add_module("relu", nn.ReLU(inplace=True))


class _BnReLU(nn.Sequential):
    BATCH_NORM = _BATCH_NORM

    def __init__(self, in_ch):
        super(_BnReLU, self).__init__()
        self.add_module("bn", _BATCH_NORM(in_ch, **_BN_KWARGS))
        self.add_module("relu", nn.ReLU(inplace=True))


class _BnReLUConv(_BnReLU):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super(_BnReLUConv, self).__init__(in_ch)
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )

#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    http://kazuto1011.github.io
# Date:   06 March 2019

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import _BnReLU, _ConvBnReLU, _Flatten

_BOTTLENECK_EXPANSION = 4


def init_weight(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def init_residual(module):
    if isinstance(module, _BasicBlockV1):
        nn.init.constant_(module.conv2.bn.weight, 0)
    elif isinstance(module, _BottleneckV1):
        nn.init.constant_(module.expand.bn.weight, 0)


class _BasicBlockV1(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super().__init__()
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, relu=False)
            if downsample
            else lambda x: x  # identity
        )
        self.conv1 = _ConvBnReLU(in_ch, out_ch, 3, stride, dilation, dilation)
        self.conv2 = _ConvBnReLU(out_ch, out_ch, 3, 1, 1, 1, relu=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h += self.shortcut(x)
        h = self.relu(h)
        return h


class _BasicBlockV2(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super().__init__()
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, stride, 0, 1, bias=False)
            if downsample
            else None  # skip preact
        )
        self.preact = _BnReLU(in_ch)
        self.conv1 = _ConvBnReLU(in_ch, out_ch, 3, stride, dilation, dilation)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, 1, bias=False)

    def forward(self, x):
        h = self.preact(x)
        s = self.shortcut(h) if self.shortcut else x
        h = self.conv1(h)
        h = self.conv2(h)
        return h + s


class _BottleneckV1(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super().__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, relu=False)
            if downsample
            else lambda x: x  # identity
        )
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, 1, 0, 1)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, stride, dilation, dilation)
        self.expand = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, relu=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.expand(h)
        h += self.shortcut(x)
        h = self.relu(h)
        return h


class _BottleneckV2(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super().__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, stride, 0, 1, bias=False)
            if downsample
            else None  # skip preact
        )
        self.preact = _BnReLU(in_ch)
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, 1, 0, 1)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, stride, dilation, dilation)
        self.expand = nn.Conv2d(mid_ch, out_ch, 1, 1, 0, 1, bias=False)

    def forward(self, x):
        h = self.preact(x)
        s = self.shortcut(h) if self.shortcut else x
        h = self.reduce(h)
        h = self.conv3x3(h)
        h = self.expand(h)
        return h + s


class _Stem(nn.Sequential):
    def __init__(self, version, in_ch, out_ch):
        super().__init__()
        conv, kwargs = {
            "v1": (_ConvBnReLU, {}),
            "v2": (nn.Conv2d, {"bias": False}),
        }.get(version)
        self.add_module("conv", conv(in_ch, out_ch, 7, 2, 3, 1, **kwargs))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1))


class _Layer(nn.Sequential):
    def __init__(self, block, n_blocks, in_ch, out_ch, stride, dilation=1, grids=None):
        super().__init__()

        # Multi-grids for DeepLab
        if grids is None:
            grids = [1 for _ in range(n_blocks)]
        else:
            assert n_blocks == len(grids)

        # Downsampling is only in the first block
        for i in range(n_blocks):
            self.add_module(
                f"block{i+1}",
                block(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class ResNetV1(nn.Sequential):
    def __init__(self, block, n_blocks, n_classes, n_filters=64):
        super().__init__()
        ch = [n_filters * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem("v1", 3, ch[0]))
        self.add_module("layer2", _Layer(block, n_blocks[0], ch[0], ch[2], 1))
        self.add_module("layer3", _Layer(block, n_blocks[1], ch[2], ch[3], 2))
        self.add_module("layer4", _Layer(block, n_blocks[2], ch[3], ch[4], 2))
        self.add_module("layer5", _Layer(block, n_blocks[3], ch[4], ch[5], 2))
        self.add_module("pool5", nn.AdaptiveAvgPool2d(1))
        self.add_module("flatten", _Flatten())
        self.add_module("fc", nn.Linear(ch[5], n_classes))
        self.apply(init_weight)
        self.apply(init_residual)


class ResNetV2(nn.Sequential):
    def __init__(self, block, n_blocks, n_classes, n_filters=64):
        super().__init__()
        ch = [n_filters * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem("v2", 3, ch[0]))
        self.add_module("layer2", _Layer(block, n_blocks[0], ch[0], ch[2], 1))
        self.add_module("layer3", _Layer(block, n_blocks[1], ch[2], ch[3], 2))
        self.add_module("layer4", _Layer(block, n_blocks[2], ch[3], ch[4], 2))
        self.add_module("layer5", _Layer(block, n_blocks[3], ch[4], ch[5], 2))
        self.add_module("postnorm", _BnReLU(ch[5]))
        self.add_module("pool5", nn.AdaptiveAvgPool2d(1))
        self.add_module("flatten", _Flatten())
        self.add_module("fc", nn.Linear(ch[5], n_classes))
        self.apply(init_weight)


def resnet18(n_classes, version="v1", **kwargs):
    ResNet, block = {
        "v1": (ResNetV1, _BasicBlockV1),
        "v2": (ResNetV2, _BasicBlockV2),
    }.get(version)
    return ResNet(block, [2, 2, 2, 2], n_classes, **kwargs)


def resnet34(n_classes, version="v1", **kwargs):
    ResNet, block = {
        "v1": (ResNetV1, _BasicBlockV1),
        "v2": (ResNetV2, _BasicBlockV2),
    }.get(version)
    return ResNet(block, [3, 4, 6, 3], n_classes, **kwargs)


def resnet50(n_classes, version="v1", **kwargs):
    ResNet, block = {
        "v1": (ResNetV1, _BottleneckV1),
        "v2": (ResNetV2, _BottleneckV2),
    }.get(version)
    return ResNet(block, [3, 4, 6, 3], n_classes, **kwargs)


def resnet101(n_classes, version="v1", **kwargs):
    ResNet, block = {
        "v1": (ResNetV1, _BottleneckV1),
        "v2": (ResNetV2, _BottleneckV2),
    }.get(version)
    return ResNet(block, [3, 4, 23, 3], n_classes, **kwargs)


def resnet152(n_classes, version="v1", **kwargs):
    ResNet, block = {
        "v1": (ResNetV1, _BottleneckV1),
        "v2": (ResNetV2, _BottleneckV2),
    }.get(version)
    return ResNet(block, [3, 8, 36, 3], n_classes, **kwargs)


if __name__ == "__main__":
    import itertools

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    image = torch.randn(1, 3, 224, 224).to(device)
    layers = [18, 34, 50, 101, 152]
    versions = ["v1", "v2"]

    for n, v in itertools.product(layers, versions):
        print(f"ResNet-{n}", v)
        model = eval(f"resnet{n}")(n_classes=1001, version=v)
        model.eval()
        model.to(device)
        print("Image:", tuple(image.shape))
        print("Logit:", tuple(model(image).shape))

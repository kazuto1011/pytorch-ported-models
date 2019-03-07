#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    http://kazuto1011.github.io
# Date:   06 March 2019

import math
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from tensorflow import keras
from torch.nn import init

from .. import modules
from ..modules import _ConvBnReLU, _Flatten, _SeparableConv2d, _SepConvBnReLU

modules._BN_KWARGS["eps"] = 1e-3
modules._BN_KWARGS["momentum"] = 0.99
_N_MIDDLES = 8


class _Block(nn.Module):
    def __init__(self, in_ch, out_ch, reps, stride, relu_first, grow_first):
        super(_Block, self).__init__()

        self.skip = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, relu=False)
            if out_ch != in_ch or stride != 1
            else lambda x: x
        )

        mid_ch = out_ch if grow_first else in_ch

        self.main = nn.Sequential()
        if relu_first:
            self.main.add_module("relu1", nn.ReLU())
        self.main.add_module("conv1", _SepConvBnReLU(in_ch, mid_ch, 3, 1, 1, 1, False))
        for i in range(reps - 2):
            self.main.add_module(f"relu{i+2}", nn.ReLU())
            self.main.add_module(
                f"conv{i+2}", _SepConvBnReLU(mid_ch, mid_ch, 3, 1, 1, 1, False)
            )
        self.main.add_module(f"relu{reps}", nn.ReLU())
        self.main.add_module(
            f"conv{reps}", _SepConvBnReLU(mid_ch, out_ch, 3, 1, 1, 1, relu=False)
        )
        if stride != 1:
            self.main.add_module("pool", nn.MaxPool2d(3, stride, 1))

    def forward(self, x):
        h = self.main(x) + self.skip(x)
        return h


class XceptionV1(nn.Sequential):
    def __init__(self, n_classes=1000):
        super(XceptionV1, self).__init__()

        self.n_classes = n_classes

        entry_layers = [
            ("conv1", _ConvBnReLU(3, 32, 3, 2, 1, 1)),
            ("conv2", _ConvBnReLU(32, 64, 3, 1, 1, 1)),
            ("block1", _Block(64, 128, 2, 2, False, True)),
            ("block2", _Block(128, 256, 2, 2, True, True)),
            ("block3", _Block(256, 728, 2, 2, True, True)),
        ]

        middle_layers = [
            (f"block{i+4}", _Block(728, 728, 3, 1, True, True))
            for i in range(_N_MIDDLES)
        ]

        exit_layers = [
            (f"block{_N_MIDDLES+4}", _Block(728, 1024, 2, 2, True, False)),
            ("conv3", _SepConvBnReLU(1024, 1536, 3, 1, 1, 1)),
            ("conv4", _SepConvBnReLU(1536, 2048, 3, 1, 1, 1)),
            ("pool", nn.AdaptiveAvgPool2d(1)),
            ("flatten", _Flatten()),
            ("fc", nn.Linear(2048, n_classes)),
        ]

        self.add_module("entry_flow", nn.Sequential(OrderedDict(entry_layers)))
        self.add_module("middle_flow", nn.Sequential(OrderedDict(middle_layers)))
        self.add_module("exit_flow", nn.Sequential(OrderedDict(exit_layers)))

    def load_from_keras(self):

        assert self.n_classes == 1000

        weights_path = keras.utils.get_file(
            "xception_weights_tf_dim_ordering_tf_kernels.h5",
            (
                "https://github.com/fchollet/deep-learning-models/"
                "releases/download/v0.4/"
                "xception_weights_tf_dim_ordering_tf_kernels.h5"
            ),
            cache_subdir="models",
            file_hash="0a58e3b7378bc2990ea3b43d5981f1f6",
        )

        with h5py.File(weights_path, mode="r") as f:
            state_dict = OrderedDict()

            s_idx = 1
            b_idx = 1
            for pth_layer, pth_module in self.named_modules():
                if isinstance(pth_module, _SeparableConv2d):
                    h5_layer = f"separableconvolution2d_{s_idx}"
                    h5_value = np.asarray(f[h5_layer][f"{h5_layer}_depthwise_kernel:0"])
                    h5_value = torch.tensor(h5_value.transpose(2, 3, 0, 1))
                    state_dict[pth_layer + ".depthwise.weight"] = h5_value
                    h5_value = np.asarray(f[h5_layer][f"{h5_layer}_pointwise_kernel:0"])
                    h5_value = torch.tensor(h5_value.transpose(3, 2, 0, 1))
                    state_dict[pth_layer + ".pointwise.weight"] = h5_value
                    s_idx += 1

                if isinstance(pth_module, nn.BatchNorm2d):
                    h5_layer = f"batchnormalization_{b_idx}"
                    h5_value = np.asarray(f[h5_layer][f"{h5_layer}_gamma:0"])
                    state_dict[pth_layer + ".weight"] = torch.tensor(h5_value)
                    h5_value = np.asarray(f[h5_layer][f"{h5_layer}_beta:0"])
                    state_dict[pth_layer + ".bias"] = torch.tensor(h5_value)
                    h5_value = np.asarray(f[h5_layer][f"{h5_layer}_running_mean:0"])
                    state_dict[pth_layer + ".running_mean"] = torch.tensor(h5_value)
                    h5_value = np.asarray(f[h5_layer][f"{h5_layer}_running_std:0"])
                    state_dict[pth_layer + ".running_var"] = torch.tensor(h5_value)
                    b_idx += 1

            idx = 1
            for pth_layer, pth_module in self.named_modules():
                if isinstance(pth_module, nn.Conv2d):
                    if pth_layer + ".weight" not in state_dict:
                        h5_layer = f"convolution2d_{idx}"
                        h5_value = np.asarray(f[h5_layer][f"{h5_layer}_W:0"])
                        h5_value = torch.tensor(h5_value.transpose(3, 2, 0, 1))
                        state_dict[pth_layer + ".weight"] = h5_value
                        idx += 1

            h5_value = torch.tensor(np.asarray(f["dense_2"]["dense_2_W:0"]))
            state_dict["exit_flow.fc.weight"] = h5_value.permute(1, 0)
            h5_value = torch.tensor(np.asarray(f["dense_2"]["dense_2_b:0"]))
            state_dict["exit_flow.fc.bias"] = h5_value

            self.load_state_dict(state_dict)


if __name__ == "__main__":
    model = XceptionV1(n_classes=1000)
    model.eval()
    model.load_from_keras()

    image = torch.randn(1, 3, 299, 299)

    print("[test]")
    print("input:", tuple(image.shape))
    print("logit:", tuple(model(image).shape))


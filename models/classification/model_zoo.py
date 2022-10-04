#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   04 March 2019

import hashlib
import os
import shutil
import tarfile
import tempfile
from collections import OrderedDict
from glob import glob
from urllib.request import urlopen

import click
import h5py
import numpy as np
import tensorflow
import torch
import torch.nn as nn
from tensorflow import keras
from tqdm import tqdm

from .modules import _ConvBnReLU, _Flatten, _SeparableConv2d, _SepConvBnReLU
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def _parse_tensorflow_ckpt(model_path):
    reader = tensorflow.train.NewCheckpointReader(model_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    data = dict()
    for name in var_to_shape_map:
        if "global_step" not in name.lower():
            tensor = reader.get_tensor(name)
            data[name] = tensor
    return data


def load_tensorflow_resnet(tar_url, ckpt_relpath, model_torch, model_name):

    torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
    model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch_home, "models"))
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(model_dir, model_name)
    if os.path.exists(save_path):
        return torch.load(save_path)

    # Download TensorFlow model
    u = urlopen(tar_url)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    else:
        file_size = None
    with tempfile.NamedTemporaryFile() as f:
        print(f"Downloading: {tar_url}")
        with tqdm(total=file_size) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        with tempfile.TemporaryDirectory() as tar_dest:
            with tarfile.open(f.name, "r:gz") as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, tar_dest)
            state_dict_tf = _parse_tensorflow_ckpt(os.path.join(tar_dest, ckpt_relpath))

    # Parse tensorflow model
    layers_tf = {"conv": set(), "bn": set(), "fc": set()}
    for name in state_dict_tf:
        name = name.split("/")
        name, param = "/".join(name[:-1]), name[-1]
        if "conv" in name:
            layers_tf["conv"].add(name)
        elif "batch_normalization" in name:
            layers_tf["bn"].add(name)
        elif "dense" in name:
            layers_tf["fc"].add(name)

    # Parse PyTorch model
    ref_state_dict = model_torch.state_dict()
    layers_torch = {"conv": [], "bn": [], "fc": []}
    for name, module in model_torch.named_modules():
        if isinstance(module, nn.Conv2d):
            layers_torch["conv"].append(name)
        elif isinstance(module, nn.BatchNorm2d):
            layers_torch["bn"].append(name)
        elif isinstance(module, SynchronizedBatchNorm2d):
            layers_torch["bn"].append(name)
        elif isinstance(module, nn.Linear):
            layers_torch["fc"].append(name)

    assert len(layers_tf["conv"]) == len(layers_torch["conv"])
    assert len(layers_tf["bn"]) == len(layers_torch["bn"])
    assert len(layers_tf["fc"]) == len(layers_torch["fc"])

    # Generate
    key_pairs = []
    add_suffix = lambda idx: f"_{idx}" if idx != 0 else ""
    for idx, key_torch in enumerate(layers_torch["conv"]):
        key_tf = "resnet_model/conv2d" + add_suffix(idx)
        key_pairs.append((key_torch + ".weight", key_tf + "/kernel"))
    for idx, key_torch in enumerate(layers_torch["bn"]):
        key_tf = "resnet_model/batch_normalization" + add_suffix(idx)
        key_pairs.append((key_torch + ".weight", key_tf + "/gamma"))
        key_pairs.append((key_torch + ".running_mean", key_tf + "/moving_mean"))
        key_pairs.append((key_torch + ".bias", key_tf + "/beta"))
        key_pairs.append((key_torch + ".running_var", key_tf + "/moving_variance"))
    for idx, key_torch in enumerate(layers_torch["fc"]):
        key_tf = "resnet_model/dense" + add_suffix(idx)
        key_pairs.append((key_torch + ".weight", key_tf + "/kernel"))
        key_pairs.append((key_torch + ".bias", key_tf + "/bias"))

    # Convert weights
    new_state_dict = OrderedDict()
    for key_torch, key_tf in key_pairs:
        param_pth = ref_state_dict[key_torch]
        param_tf = state_dict_tf[key_tf]
        if param_tf.ndim == 2:
            param_tf = param_tf.transpose(1, 0)
        elif param_tf.ndim == 4:
            param_tf = param_tf.transpose(3, 2, 0, 1)
        assert (
            tuple(param_pth.shape) == param_tf.shape
        ), "Inconsistent shape: {}, {}".format(tuple(param_pth.shape), param_tf.shape)
        new_state_dict[key_torch] = torch.FloatTensor(param_tf)

    model_torch.load_state_dict(new_state_dict)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(new_state_dict, save_path)

    return new_state_dict


def load_keras_xceptionv1(model_torch):

    assert model_torch.n_classes == 1000

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
        for pth_layer, pth_module in model_torch.named_modules():
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
            elif isinstance(pth_module, SynchronizedBatchNorm2d):
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
        for pth_layer, pth_module in model_torch.named_modules():
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

        return state_dict


def load_keras_model():
    pass

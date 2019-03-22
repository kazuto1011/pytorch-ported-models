#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 March 2019


def xception_v1(pretrained=False, *args, **kwargs):
    """
    Xception v1 model
    """

    from models.classification.xception import xception_v1 as _xception_v1

    model = _xception_v1(pretrained=True, *args, **kwargs)

    return model

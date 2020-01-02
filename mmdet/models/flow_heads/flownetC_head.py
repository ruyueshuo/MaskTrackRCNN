#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 下午2:04
# @Author  : FengDa
# @File    : flownetC_head.py
# @Software: PyCharm

import logging
import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from .flow_utils import conv, predict_flow, deconv, crop_like, correlate
from .flownetC import FlowNetC
from ..registry import HEADS


@HEADS.register_module
class FlowNetCHead(FlowNetC):

    def __init__(self):
        super(FlowNetCHead, self).__init__()

    def forward(self, x, out_conv2a):
        """

        :param x: feature map (48, 64, 512)
        :return:flow
        """
        # x1 = x[:, :3]
        # x2 = x[:, 3:]

        # out_conv1a = self.conv1(x1)
        # out_conv2a = self.conv2(out_conv1a)
        # out_conv3a = self.conv3(out_conv2a)
        #
        # out_conv1b = self.conv1(x2)
        # out_conv2b = self.conv2(out_conv1b)
        # out_conv3b = self.conv3(out_conv2b)
        out_conv3a = x[:, : 256]
        out_conv3b = x[:, 256:]
        # out_conv2a = x

        out_conv_redir = self.conv_redir(out_conv3a)
        out_correlation = correlate(out_conv3a, out_conv3b)

        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)

        out_conv3 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2a)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2a)

        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        # elif pretrained is None:
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv2d):
        #             kaiming_init(m)
        #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #             constant_init(m, 1)
        #
        #     if self.zero_init_residual:
        #         for m in self.modules():
        #             if isinstance(m, Bottleneck):
        #                 constant_init(m.norm3, 0)
        #             elif isinstance(m, BasicBlock):
        #                 constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

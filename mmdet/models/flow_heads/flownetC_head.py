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
from .correlation_package.correlation import Correlation
from .propagation_utils import warp

@HEADS.register_module
class FlowNetCHead(FlowNetC):

    def __init__(self, checkpoint=None):
        super(FlowNetCHead, self).__init__()
        if checkpoint:
            self.checkpoint = checkpoint

        # Correlation function in FlowNet2 by Nvidia.
        # See https://github.com/NVIDIA/flownet2-pytorch for details.
        self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x, y):
        """
        :param x: tensor, feature map of low resolution current frame (b, 256, 48, 64)
        :param y: list, feature map of high resolution key frame (b, 256, 48, 64)
                  and resnet out0 of high resolution key frame (b, 128, 96, 128)
        :return:flow: tensor, (b, 2, 96, 128)
        """
        out_conv2a = y[0]
        out_conv3a = y[1]
        out_conv3b = x

        assert out_conv3a.shape == out_conv3b.shape

        out_conv_redir = self.conv_redir(out_conv3a)
        # out_correlation = correlate(out_conv3a, out_conv3b)
        out_corr = self.corr(out_conv3a, out_conv3b)  # False
        out_correlation = self.corr_activation(out_corr)

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

    def init_weights(self):
        """load checkpoint."""
        load_checkpoint(self, self.checkpoint)

    @staticmethod
    def flow2rgb(flow_map, max_value):
        """flow to rgb images files"""
        flow_map_np = flow_map.detach().cpu().numpy()
        _, h, w = flow_map_np.shape
        flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
        rgb_map = np.ones((3, h, w)).astype(np.float32)
        if max_value is not None:
            normalized_flow_map = flow_map_np / max_value
        else:
            normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
        rgb_map[0] += normalized_flow_map[0]
        rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
        rgb_map[2] += normalized_flow_map[1]
        return rgb_map.clip(0, 1)

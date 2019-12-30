#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 上午11:18
# @Author  : FengDa
# @File    : res5_mask_head.py
# @Software: PyCharm
import logging
import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from mmdet.models.mask_heads.resnet import conv1x1, BasicBlock, Bottleneck
from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import mask_cross_entropy, mask_target


@HEADS.register_module
class ResMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 normalize=None,
                 # resnet
                 norm_eval=True,
                 layer=3,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 block=Bottleneck,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResMaskHead, self).__init__()
        # Head net init
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.normalize = normalize
        self.with_bias = normalize is None

        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                # self.conv_out_channels,
                self.conv_out_channels * block.expansion,  # expansion: res5 channel expansion in Bottleneck
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.conv_logits = nn.Conv2d(self.conv_out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

        # ResNet Init
        self.norm_eval = norm_eval
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = in_channels
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(block, in_channels, layer, stride=1,
                                       dilate=replace_stride_with_dilation)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Make residual net layer.
        Set stride=1 according to MaskR-CNN to ensure that the output size is (c, 28, 28)."""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        # make residual layer
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        """Initialize weights."""
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.layer4:
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)
        # if isinstance(pretrained, str):
        #     logger = logging.getLogger()
        #     load_checkpoint(self, pretrained, strict=False, logger=logger)
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
        # else:
        #     raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward. res5->upsample->mask"""
        x = self.layer4(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def train(self, mode=True):
        """To ensure that BN doesn't train."""
        super(ResMaskHead, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = mask_cross_entropy(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
        else:
            loss_mask = mask_cross_entropy(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale, det_obj_ids=None):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        if det_obj_ids is not None:
            obj_segms = {}
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            if det_obj_ids is not None:
                if det_obj_ids[i] >= 0:
                    obj_segms[det_obj_ids[i]] = rle
            else:
                cls_segms[label - 1].append(rle)
        if det_obj_ids is not None:
            return obj_segms
        else:
            return cls_segms


if __name__=="__main__":
    # load part of pre trained model
    """
    pretrained_dict = ...
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    """

    model = ResMaskHead()
    print(model)
    print("finish")
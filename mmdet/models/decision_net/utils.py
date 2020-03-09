#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/19 上午9:50
# @Author  : FengDa
# @File    : utils.py
# @Software: PyCharm
import numpy as np

import torch
import torch.nn as nn
import mmcv
import torchvision.models as models
from mmdet.datasets import build_dataloader
from mmdet.datasets.transforms import MaskTransform


def trans_action(a):
    if a >= 0.75:
        a = 0
    elif a >= 0.5:
        a = 1
    elif a >= 0.25:
        a = 2
    else:
        a = 3
    return a


def change_img_meta(data, scale_factor):
    # resize image according to scale factor.
    img = data['img'][0]
    img_meta = data['img_meta'][0]

    # change image meta info according to scale factor change.
    img_meta[0][0]['img_shape'] = (int(img_meta[0][0]['img_shape'][0] * scale_factor),
                                   int(img_meta[0][0]['img_shape'][1] * scale_factor), 3)
    img_meta[0][0]['pad_shape'] = img.shape
    img_meta[0][0]['scale_factor'] = img_meta[0][0]['scale_factor'] * scale_factor

    data['img_meta'][0] = img_meta

    return data


def resize(img, scale_factor, size_divisor=None):
    # resize image according to scale factor.
    if isinstance(scale_factor, tuple):
        img = nn.functional.interpolate(img, scale_factor, mode='bilinear', align_corners=True)
    else:
        img = nn.functional.interpolate(img, scale_factor=scale_factor, mode='bilinear', align_corners=True)

    if size_divisor:
        device = img.device
        img_np = np.transpose(img.squeeze().detach().cpu().numpy(), (1, 2, 0))

        # The size of last feature map is 1/32(specially for resnet50) of the original image,
        # Impad to ensure that the original image size is multiple of 32.
        img_np = mmcv.impad_to_multiple(img_np, size_divisor)

        img = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)
        # data['img'][0] = img

    return img


def modify_cfg(cfg, video_name, is_train=True):
    cfg_test = cfg.data.val
    if is_train:
        folder = 'train'
    else:
        folder = 'valid'
    cfg_test.ann_file = cfg.data_root + '{}/Annotations/{}/{}_annotations.json'.format(folder, video_name, video_name)
    cfg_test.img_prefix = cfg.data_root + '{}/JPEGImages/'.format(folder)
    return cfg_test


def get_dataloader(dataset):
    # dataset = obj_from_dict(cfg, datasets, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False)
    return data_loader


def get_VIS_data(loader, index, dataset=None, size=(384, 640)):
    data = dataset[index]
    return container_to_tensor(data, size=size)


def container_to_tensor(data, size):
    data['img'] = data['img'].data.cuda()
    data['ref_img'] = data['ref_img'].data.cuda()
    if len(data['img'].shape) != 4:
        data['img'] = data['img'].unsqueeze(0)
    if len(data['ref_img'].shape) != 4:
        data['ref_img'] = data['ref_img'].unsqueeze(0)

    if data['img'].shape[-2:] != size:
        data['img'] = nn.functional.interpolate(data['img'], size, mode='bilinear', align_corners=True)
        data['ref_img'] = nn.functional.interpolate(data['ref_img'], size, mode='bilinear', align_corners=True)

    data['img_meta'] = data['img_meta'].data
    data['gt_bboxes'] = data['gt_bboxes'].data.cuda()
    data['ref_bboxes'] = data['ref_bboxes'].data.cuda()
    data['gt_labels'] = data['gt_labels'].data.cuda()
    data['gt_pids'] = data['gt_pids'].data.cuda()
    data['gt_bboxes_ignore'] = data['gt_bboxes_ignore'].data.cuda()
    data['gt_masks'] = [data['gt_masks'].data]

    img_meta = []
    img_meta.append(data['img_meta'])
    data['img_meta'] = img_meta
    data['gt_bboxes'] = [data['gt_bboxes']]
    data['ref_bboxes'] = [data['ref_bboxes']]
    data['gt_labels'] = [data['gt_labels']]
    data['gt_pids'] = [data['gt_pids']]
    data['gt_bboxes_ignore'] = [data['gt_bboxes_ignore']]
    return data


def get_low_data(data_current_full, scale_facotr, size_divisor=32, is_train=True):
    data_current_a = data_current_full.copy()
    if is_train:
        data_current_a['img'] = resize(data_current_a['img'], scale_factor=scale_facotr, size_divisor=size_divisor)
        data_current_a['ref_img'] = resize(data_current_a['ref_img'], scale_factor=scale_facotr,
                                           size_divisor=size_divisor)
        data_current_a['gt_bboxes'][0] = (data_current_a['gt_bboxes'][0] * scale_facotr).floor()
        data_current_a['ref_bboxes'][0] = (data_current_a['ref_bboxes'][0] * scale_facotr).floor()
        mask_transform = MaskTransform()
        h, w = data_current_a['img'].shape[-2:]
        data_current_a['gt_masks'][0] = mask_transform(data_current_a['gt_masks'][0], pad_shape=(h, w),
                                                       scale_factor=scale_facotr, flip=False)
        # cv2.resize(self.data_current_low['gt_masks'][0], fx=scale_facotr, fy, size_divisor=32)
        data_current_a['img_meta'][0]['img_shape'] = \
            (int(data_current_a['img_meta'][0]['img_shape'][0]*scale_facotr),
             int(data_current_a['img_meta'][0]['img_shape'][1]*scale_facotr), 3)
        data_current_a['img_meta'][0]['pad_shape'] = \
            (data_current_a['img'][0].shape[-2], data_current_a['img'][0].shape[-1], 3)
        data_current_a['img_meta'][0]['scale_factor'] *= scale_facotr
    else:
        data_current_a['img'][0] = resize(data_current_a['img'][0], scale_factor=scale_facotr, size_divisor=size_divisor)
        data_current_a['img_meta'][0].data[0][0]['img_shape'] = \
            (int(data_current_a['img_meta'][0].data[0][0]['img_shape'][0]*scale_facotr),
             int(data_current_a['img_meta'][0].data[0][0]['img_shape'][1]*scale_facotr), 3)
        data_current_a['img_meta'][0].data[0][0]['pad_shape'] = \
            (data_current_a['img'][0].shape[-2], data_current_a['img'][0].shape[-1], 3)
        data_current_a['img_meta'][0].data[0][0]['scale_factor'] *= scale_facotr
    return data_current_a


def one_hot(a):
    if a == 0:
        return np.array([0, 0])
    elif a == 1:
        return np.array([0, 1])
    elif a == 2:
        return np.array([1, 0])
    elif a == 3:
        return np.array([1, 1])



def get_self_feat(model, input, feat_size=(24, 40)):
    '''
    :param self:
    :param img_tensor: a tensor with size of batchsize * channels * height * weight
    :return:
    '''
    outs, out0 = model.extract_feat(input)
    if outs[0].shape[-2:] != feat_size:
        return resize(outs[0], feat_size)
    else:
        return outs[0]


# @staticmethod
def get_diff_feat(full_data, current_data, feat_size=(24, 40)):
    if full_data.shape[-2:] != feat_size:
        full_data = resize(full_data, feat_size)
    if current_data.shape[-2:] != feat_size:
        current_data = resize(current_data, feat_size)
    # full_data['img'][0] = self.resize(full_data['img'][0], scale_factor=self.actions[-1])
    assert full_data.shape == current_data.shape, "feature size donnot match."
    return full_data - current_data
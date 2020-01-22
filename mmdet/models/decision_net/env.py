#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/14 下午1:38
# @Author  : FengDa
# @File    : env.py
# @Software: PyCharm
import os
import random
import json
from queue import Queue
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from mmdet import datasets
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmdet.datasets import get_dataset
from mmdet.datasets import build_dataloader
from .memory import ReplayMemory

from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval


def modify_cfg(cfg, video_name):
    cfg_test = cfg.data.val
    cfg_test.ann_file = cfg.data_root + 'train/Annotations/{}/{}_annotations.json'.format(video_name, video_name)
    cfg_test.img_prefix = cfg.data_root + 'train/JPEGImages/'
    return cfg_test


def get_dataloader(dataset):
    # dataset = obj_from_dict(cfg, datasets, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False)
        # shuffle=False)
    return data_loader


def get_VIS_data(loader, index):
    for i, data in enumerate(loader):
        if i == index:
            # TODO DC objects to GPU tensor.
            data['img'] = data['img'].data[0].cuda()
            data['ref_img'] = data['ref_img'].data[0].cuda()
            data['img_meta'] = data['img_meta'].data[0]
            data['gt_bboxes'] = data['gt_bboxes'].data[0]
            data['ref_bboxes'] = data['ref_bboxes'].data[0]
            data['gt_labels'] = data['gt_labels'].data[0]
            data['gt_pids'] = data['gt_pids'].data[0]
            data['gt_bboxes_ignore'] = data['gt_bboxes_ignore'].data[0]
            data['gt_masks'] = data['gt_masks'].data[0]

            data['gt_bboxes'][0] = data['gt_bboxes'][0].cuda()
            data['ref_bboxes'][0] = data['ref_bboxes'][0].cuda()
            data['gt_labels'][0] = data['gt_labels'][0].cuda()
            data['gt_pids'][0] = data['gt_pids'][0].cuda()
            data['gt_bboxes_ignore'][0] = data['gt_bboxes_ignore'][0].cuda()
            return data

    raise LookupError


class DecisionEnv(object):
    """env for decision."""

    def __init__(self,
                 model,
                 cfg,
                 ):

        self.model = model  # task model
        self.cfg = cfg  # config

        self.state = None
        self.rewards = None
        self.idx_frame = None    # frame index of the episode.
        self.start_frame = None  # first frame index of the original video
        self.done = None

        # features
        self.feat_diff = None
        self.feat_self = None
        self.feat_FAR = None
        # self.feat_history = Queue(maxsize=5)
        self.feat_history = None

        # temp variables
        self.dataset = None
        self.data_loader = None
        self.feat_last_full = None
        self.feat_map_last = None
        self.data_last_full = None
        # self.current_low_feat = None
        self.data_current_full = None
        self.data_current_low = None

        self.cfg_test = None
        self.video_annotation = None

        data_path = cfg['data_root'] + 'train/Annotations'
        self.videos = os.listdir(data_path)  # video names
        self.videos.sort()
        self.actions = [1, 1/2, 1/3, 1/4]

        self.device = torch.device("cuda")

    def _state_reset(self):

        # self.data = self.data_loader[self.start_frame + self.idx_frame]
        # self.data = get_VIS_data(self.data_loader, self.start_frame + self.idx_frame)
        self.data_current_full = get_VIS_data(self.data_loader, self.start_frame + self.idx_frame)
        self.data_last_full = self.data_current_full
        self.feat_last_full = self.get_self_feat(self.data_current_full['img'])

        # self.current_data['img'][0] = self.resize(self.current_data['img'][0], scale_factor=self.actions[-1])
        # 自身的特征
        self.feat_self = self.resize(self.feat_last_full, scale_factor=self.actions[-1])
        # 跟高清差异的特征
        self.feat_diff = torch.zeros_like(self.feat_self)

        self.feat_FAR = torch.tensor([0]).cuda()
        self.feat_history = torch.zeros([5]).cuda()
        # for i in range(self.feat_history.maxsize):
        #     self.feat_history.put(0)

        self.state = [self.feat_diff, self.feat_self, self.feat_FAR, self.feat_history]

    def reset(self):

        # random select a video
        # video_id = np.random.randint(len(self.videos))
        # video_name = self.videos[video_id]
        video_name = random.sample(self.videos, 1)[0]
        print('video name: {}.'.format(video_name))
        # get data loader of the selected video
        self.cfg_test = modify_cfg(self.cfg, video_name)
        # self.dataset = obj_from_dict(self.cfg_test, datasets, dict(test_mode=True))
        self.dataset = get_dataset(self.cfg_test)
        print(len(self.dataset))
        self.data_loader = get_dataloader(self.dataset)

        # random choose a start frame, at list 10 frames to run.
        # print('len of loader ', len(self.data_loader))
        self.start_frame = np.random.randint(min(len(self.data_loader) - 10, 10))
        self.idx_frame = 0

        # reset state
        self._state_reset()

        self.rewards = []
        self.done = False
        return self.state

    def _update_state(self, a):

        # next frame == current frame
        self.idx_frame += 1
        if a in [0]:
            # 动作选取为高清图片时， full_img和feat_FAR需要相应更改。
            self.data_last_full = get_VIS_data(self.data_loader, self.start_frame + self.idx_frame - 1)
            self.feat_FAR = (self.feat_FAR * self.idx_frame + 1) / (self.idx_frame + 1)
            self.feat_last_full = self.get_self_feat(self.data_last_full['img'])
        else:
            self.feat_FAR = (self.feat_FAR * self.idx_frame + 0) / (self.idx_frame + 1)

        self.data_current_full = get_VIS_data(self.data_loader, self.start_frame + self.idx_frame)
        self.data_current_low = self.data_current_full

        self.data_current_low['img'] = self.resize(self.data_current_low['img'], scale_factor=self.actions[-1])
        self.data_current_low['ref_img'] = self.resize(self.data_current_low['ref_img'], scale_factor=self.actions[-1])
        self.data_current_low['gt_bboxes'][0] = (self.data_current_low['gt_bboxes'][0] * self.actions[-1]).floor()
        self.data_current_low['ref_bboxes'][0] = (self.data_current_low['ref_bboxes'][0] * self.actions[-1]).floor()
        # self.data_current_low['gt_masks'][0] = self.resize(self.data_current_low['gt_masks'][0], scale_factor=self.actions[-1])
        self.data_current_low['img_meta'][0]['img_shape'] = (180, 320, 3)
        self.data_current_low['img_meta'][0]['pad_shape'] = (96, 160, 3)
        self.data_current_low['img_meta'][0]['scale_factor'] = 0.125
        # 自身的特征
        self.feat_self = self.get_self_feat(self.data_current_low['img'])

        # 跟高清差异的特征
        feat_last_low = self.resize(self.feat_last_full, scale_factor=self.actions[-1])
        self.feat_diff = self.get_diff_feat(feat_last_low, self.feat_self)

        # 历史动作特征
        self.feat_history[:-1] = self.feat_history[1:]
        self.feat_history[-1] = a

        self.state = [self.feat_diff, self.feat_self, self.feat_FAR, self.feat_history]
        return self.state

    def _get_reward(self, a):

        # When action is full resolution. Update last full feature maps, return reward 0.
        if a in [0]:
            data = self.data_last_full
            with torch.no_grad():
                _, feat_map = self.model(return_loss=True, key_feat=None, **data)
            self.feat_map_last = feat_map
            return 0

        # When action is low resolution.
        # Calculate the loss of full resolution and low resolution image data, return reward.
        key_feat = self.data_last_full
        key_feat['feat_map_last'] = self.feat_map_last
        # with torch.no_grad():
        loss_full, _ = self.model(return_loss=True, key_feat=None, **self.data_current_full)

        loss_low, _ = self.model(return_loss=True, key_feat=key_feat, **self.data_current_low)

        r = loss_low['loss_mask'] - loss_full['loss_mask']

        return r

    def step(self, a):
        a = self.trans_action(a)
        assert a in [0, 1, 2, 3]

        s_plus = self._update_state(a)

        r = self._get_reward(a)

        FAR_thr = 0.8
        if self.feat_FAR > FAR_thr:
            self.done = True

        if (self.start_frame + self.idx_frame) >= len(self.dataset)-1:
            self.done = True

        return s_plus, r, self.done

    @staticmethod
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

    def render(self):
        pass

    def select_action(self):
        pass

    def seed(self, s):
        pass

    @staticmethod
    def resize(img, scale_factor, size_divisor=None):
        # resize image according to scale factor.

        img = nn.functional.interpolate(img, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        device = img.device
        img_np = np.transpose(img.squeeze().detach().cpu().numpy(), (1, 2, 0))

        if size_divisor:
            # The size of last feature map is 1/32(specially for resnet50) of the original image,
            # Impad to ensure that the original image size is multiple of 32.
            img_np = mmcv.impad_to_multiple(img_np, size_divisor)

        img = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)
        # data['img'][0] = img

        return img

    @staticmethod
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

    # @staticmethod
    def get_self_feat(self, input):
        '''
        :param self:
        :param img_tensor: a tensor with size of batchsize * channels * height * weight
        :return:
        '''
        model = models.resnet50(pretrained=True).to(self.device)
        model.eval()

        x = input
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        # print(x.size())

        return x.cuda()

    @staticmethod
    def get_diff_feat(full_data, current_data):
        # full_data['img'][0] = self.resize(full_data['img'][0], scale_factor=self.actions[-1])
        return full_data - current_data

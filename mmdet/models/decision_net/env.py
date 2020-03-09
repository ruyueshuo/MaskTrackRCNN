#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/14 下午1:38
# @Author  : FengDa
# @File    : env.py
# @Software: PyCharm
import os
import random
import traceback
import json
from queue import Queue
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import copy

import cv2
from mmdet import datasets
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmdet.datasets import get_dataset
from mmdet.datasets import build_dataloader
from .memory import ReplayMemory
from .utils import *
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval


class DecisionEnv(object):
    """Env for decision."""

    def __init__(self,
                 model,
                 cfg,
                 videos=None,
                 # res_model=None,
                 is_train=True):

        self.model = model  # task model
        self.cfg = cfg  # config
        if videos:
            self.videos = videos  # video list
        else:
            data_path = cfg['data_root'] + 'train/Annotations'
            self.videos = os.listdir(data_path)  # video names
        self.videos.sort()
        self.is_train = is_train  # train or validation

        self.state = None
        self.rewards = None
        self.idx_frame = None  # frame index of the episode.
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
        self.data_current_a = None

        self.cfg_test = None
        self.video_annotation = None

        self.actions = [1, 1/2, 1/3, 1/4]
        self.speed_rewards = [0., 0.2, 0.3, 0.4]
        # self.device = torch.device("cuda")
        self.FAR_thr = torch.tensor([0.]).cuda()
        self.feat_size = (24, 40)  # feature size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _state_reset(self):
        """Reset Features for states."""
        # self.feat_self = resize(self.feat_last_full, scale_factor=self.actions[-1])
        self.feat_self = self.feat_last_full
        self.feat_diff = torch.zeros_like(self.feat_self)
        self.feat_FAR = torch.tensor([0.]).cuda()
        self.feat_history = torch.zeros([10]).cuda()
        self.state = [self.feat_diff, self.feat_self, self.feat_FAR, self.feat_history]
        # self.state = [self.feat_diff, self.feat_self, self.feat_history]

    def reset(self, index=0):
        # if self.is_train:
        #     dataset = get_dataset(self.cfg.data.train)
        # else:
        #     dataset = get_dataset(self.cfg.data.val)
        # # prepare data loaders
        # data_loaders = [build_dataloader(dataset, self.cfg.data.imgs_per_gpu, self.cfg.data.workers_per_gpu, 1, dist=False)]

        # Build data loader by each video.
        if self.is_train:
            # Randomly select a video from train list
            video_name = random.sample(self.videos, 1)[0]
        else:
            # Select video from validation list
            video_name = self.videos[index]

        # Get data loader of the selected video
        self.cfg_test = modify_cfg(self.cfg, video_name)
        # TODO DATASET 生成方式
        # self.dataset = obj_from_dict(self.cfg_test, datasets, dict(test_mode=True))
        self.dataset = get_dataset(self.cfg_test)
        print('video name: {}.\t len of dataset{}. '.format(video_name, len(self.dataset)))
        self.data_loader = get_dataloader(self.dataset)
        if self.is_train:
            # random choose a start frame, at list 10 frames to run.
            if len(self.data_loader) <= 10:
                self.start_frame = 0
            else:
                self.start_frame = np.random.randint(min(len(self.data_loader) - 10, 10))
        else:
            # start from the first frame.
            self.start_frame = 0

        self.idx_frame = 0
        self.data_current_full = get_VIS_data(self.data_loader, self.start_frame + self.idx_frame, dataset=self.dataset)
        self.data_last_full = self.data_current_full
        self.feat_last_full = self.get_self_feat(self.data_current_full['img'])

        self._state_reset()
        self.rewards = []
        self.done = False

        return self.state

    def _update_state(self, a):

        # next frame == current frame
        self.idx_frame += 1
        if a in [0]:
            # 动作选取为高清图片时， full_img和feat_FAR需要相应更改。
            self.data_last_full = get_VIS_data(self.data_loader, self.start_frame + self.idx_frame - 1, dataset=self.dataset)
            # self.feat_FAR = (self.feat_FAR * self.idx_frame + 1) / (self.idx_frame + 1)
            self.feat_FAR = torch.tensor([0.]).cuda()
            self.feat_last_full = self.get_self_feat(self.data_last_full['img'])
        else:
            # self.feat_FAR = (self.feat_FAR * self.idx_frame + 0) / (self.idx_frame + 1)
            self.feat_FAR += 0.05
            if self.feat_FAR > 1:
                self.feat_FAR = torch.tensor([1.]).cuda()

        self.data_current_full = get_VIS_data(self.data_loader, self.start_frame + self.idx_frame, dataset=self.dataset)
        self.data_current_low = get_low_data(copy.deepcopy(self.data_current_full), scale_facotr=self.actions[-1])
        self.data_current_a3 = self.data_current_low.copy()
        self.data_current_a2 = get_low_data(copy.deepcopy(self.data_current_full), scale_facotr=self.actions[2])
        self.data_current_a1 = get_low_data(copy.deepcopy(self.data_current_full), scale_facotr=self.actions[1])

        # Update states.
        self.feat_self = self.get_self_feat(self.data_current_low['img'])
        # feat_last_low = resize(self.feat_last_full, scale_factor=self.actions[-1])
        self.feat_diff = self.get_diff_feat(self.feat_last_full, self.feat_self)

        self.feat_history[:-2] = self.feat_history[2:]
        self.feat_history[-2:] = torch.from_numpy(one_hot(a)).to(self.device)

        self.state = [self.feat_diff, self.feat_self, self.feat_FAR, self.feat_history]
        # self.state = [self.feat_diff, self.feat_self, self.feat_history]
        return self.state

    def _get_reward(self, a, show=False):
        """ When action is full resolution. Update last full feature maps, return reward 0. When action is
        low resolution. Calculate the loss of full resolution and low resolution image data, return reward."""
        with torch.no_grad():
            loss_full, feat_map = self.model(return_loss=True, key_frame=None, **self.data_current_full)
            if self.feat_map_last is None:
                return 0, feat_map
            key_feat = self.data_last_full
            key_feat['feat_map_last'] = self.feat_map_last
            loss_low3, _ = self.model(return_loss=True, key_frame=key_feat, **self.data_current_a3)
            loss_low2, _ = self.model(return_loss=True, key_frame=key_feat, **self.data_current_a2)
            loss_low1, _ = self.model(return_loss=True, key_frame=key_feat, **self.data_current_a1)

            # 去掉match loss\loss_reg\loss_cls
            loss_f = loss_full['loss_mask'] + loss_full['loss_cls'] + loss_full['loss_reg']
            loss_l3 = loss_low3['loss_mask'] + loss_low3['loss_cls'] + loss_low3['loss_reg']
            loss_l2 = loss_low2['loss_mask'] + loss_low2['loss_cls'] + loss_low2['loss_reg']
            loss_l1 = loss_low1['loss_mask'] + loss_low1['loss_cls'] + loss_low1['loss_reg']
            # print("loss_f:{},\tloss_l1:{},\tloss_l2:{},\tloss_l3:{}"
            #       .format(loss_f.item(), loss_l1.item(), loss_l2.item(), loss_l3.item()))

            loss = [loss_f.item(), loss_l1.item(), loss_l2.item(), loss_l3.item()]
            r = min(loss) - loss[a] + self.speed_rewards[a]
            # r = min(loss) - loss[a]

            # show = True
            # if show:
            #     save_path = '/home/ubuntu/code/fengda/MaskTrackRCNN/'
            #     dataset = self.data_loader.dataset
            #     result = self.model(return_loss=False, **self.data_current_full)
            #     self.model.module.show_result(self.data_current_full, result, dataset.img_norm_cfg,
            #                                   dataset=dataset.CLASSES,
            #                                   save_vis=True,
            #                                   save_path=save_path,
            #                                   is_video=True)
            # r = 0
            if a in [0]:
                return r, feat_map
            else:
                return r, None

    def step(self, a):
        # a = self.trans_action(a)
        assert a in [0, 1, 2, 3]

        self._update_state(a)

        try:
            r, feat_map = self._get_reward(a)
        except RuntimeError:
            print(traceback.print_exc())
            r = 0
            feat_map = None

        if a in [0]:
            assert feat_map is not None, "Feature maps cannot be None when action is 0."
            self.feat_map_last = feat_map
        # if self.is_train:
        #     if self.feat_FAR > self.FAR_thr:
        #         self.done = True

        if (self.start_frame + self.idx_frame) >= len(self.dataset) - 1:
            self.done = True
        print("frame:{},\taction:{},\treward:{}".format(self.start_frame + self.idx_frame, a, r))
        return self.state, r, self.done

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

    # @staticmethod
    def get_self_feat(self, input):
        '''
        :param self:
        :param img_tensor: a tensor with size of batchsize * channels * height * weight
        :return:
        '''
        outs, out0 = self.model.extract_feat(input)
        if outs[0].shape[-2:] != self.feat_size:
            return resize(outs[0], self.feat_size)
        else:
            return outs[0]

    # @staticmethod
    def get_diff_feat(self, full_data, current_data):
        if full_data.shape[-2:] != self.feat_size:
            full_data = resize(full_data, self.feat_size)
        if current_data.shape[-2:] != self.feat_size:
            current_data = resize(current_data, self.feat_size)
        # full_data['img'][0] = self.resize(full_data['img'][0], scale_factor=self.actions[-1])
        assert full_data.shape == current_data.shape, "feature size donnot match."
        return full_data - current_data




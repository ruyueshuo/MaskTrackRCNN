#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/14 下午1:38
# @Author  : FengDa
# @File    : env.py
# @Software: PyCharm
import os
import random
from queue import Queue
import numpy as np
import torch
import torch.nn as nn

from mmdet import datasets
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmdet.datasets import build_dataloader
from .memory import ReplayMemory
from pycocotools.ytvoseval import YTVOSeval


def modify_cfg(cfg, video_name):
    cfg_test = cfg.data.test
    cfg_test.ann_file = cfg.data_root + 'train/Annotations/{}/{}_annotations.json'.format(video_name, video_name)
    cfg_test.img_prefix =cfg.data_root + 'train/JPEGImages/'
    return cfg_test


def get_dataloader(dataset):
    # dataset = obj_from_dict(cfg, datasets, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=1,
        num_gpus=1,
        dist=False,
        shuffle=False)
    return data_loader


def get_VIS_data(loader, index):
    for i, data in enumerate(loader):
        if i == index:
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
        self.last_full_feat = None
        self.current_data = None

        data_path = cfg['data_root'] + 'train/Annotations'
        self.videos = os.listdir(data_path)  # video names
        self.actions = [1, 1/2, 1/3, 1/4]

    def _state_reset(self):

        # self.data = self.data_loader[self.start_frame + self.idx_frame]
        self.data = get_VIS_data(self.data_loader, self.start_frame + self.idx_frame)

        # 跟高清差异的特征
        self.feat_diff = None
        # 自身的特征
        self.feat_self = None

        self.feat_FAR = 0
        self.feat_history = np.zeros((5,))
        # for i in range(self.feat_history.maxsize):
        #     self.feat_history.put(0)

        self.state = [self.feat_diff, self.feat_self, self.feat_FAR, self.feat_history]

    def reset(self):

        # random select a video
        video_name = random.sample(self.videos, 1)[0]

        # get data loader of the selected video
        cfg_test = modify_cfg(self.cfg, video_name)
        self.dataset = obj_from_dict(cfg_test, datasets, dict(test_mode=True))
        self.data_loader = get_dataloader(self.dataset)

        # random choose a start frame, at list 10 frames to run.
        # print('len of loader ', len(self.data_loader))
        self.start_frame = np.random.randint(min(len(self.data_loader) - 10, 10))
        self.idx_frame = 0

        # reset state
        self._state_reset()

        self.last_full_feat = self.feat_self

        self.rewards = []
        self.done = False
        return self.state

    def _update_state(self, a):

        # next frame
        self.idx_frame += 1
        if a in [0]:
            # 动作选取为高清图片时， full_img和feat_FAR需要相应更改。
            self.full_img = self.data_loader[self.idx_frame - 1]
            self.feat_FAR = (self.feat_FAR * self.idx_frame + 1) / (self.idx_frame + 1)
        else:
            self.feat_FAR = (self.feat_FAR * self.idx_frame + 0) / (self.idx_frame + 1)

        full_img = self.full_img
        current_img = self.data_loader[self.idx_frame]

        # 跟高清差异的特征
        self.feat_diff = get_diff_feat(full_img , current_img)
        # 自身的特征
        self.feat_self = get_self_feat(current_img)

        self.feat_history[0:-2] = self.feat_history[1:]
        self.feat_history[-1] = a

        self.state = [self.feat_diff, self.feat_self, self.feat_FAR, self.feat_history]
        return self.state

    def _get_reward(self, a):
        if a in [0]:
            return 0

        # calculate the loss of full resolution image data
        data = self.current_data
        loss_full = self.model(return_loss=True, rescale=True, **data)

        # calculate the loss of low resolution image data according to action
        data_low = self.resize(data, self.actions[a])
        loss_low = self.model(return_loss=True, rescale=True, **data_low)

        r = loss_low - loss_full

        # # TODO reward calculation
        # if mmcv.is_str(ytvos):
        #     ytvos = YTVOS(ytvos)
        # assert isinstance(ytvos, YTVOS)
        #
        # if len(ytvos.anns) == 0:
        #     print("Annotations does not exist")
        #     return
        # assert result_file.endswith('.json')
        # ytvos_dets = ytvos.loadRes(result_file)
        #
        # vid_ids = ytvos.getVidIds()
        # for res_type in result_types:
        #     iou_type = res_type
        #     ytvosEval = YTVOSeval(ytvos, ytvos_dets, iou_type)
        #     ytvosEval.params.vidIds = vid_ids
        #     if res_type == 'proposal':
        #         ytvosEval.params.useCats = 0
        #         ytvosEval.params.maxDets = list(max_dets)
        #     ytvosEval.evaluate()
        #     ytvosEval.accumulate()
        #     ytvosEval.summarize()
        # data = self.current_data
        # result = model(return_loss=False, rescale=True, **data)
        # ytvos_eval(result, eval_types, self.dataset.ytvos)
        # r = a * self.state
        return r

    def step(self, a):
        assert a in [0, 1, 2, 3]

        # s = self.state

        # a = self.policy(self.state)

        s_plus = self._update_state(a)

        r = self._get_reward(a)

        FAR_thr = 0.6
        if self.feat_FAR > FAR_thr:
            self.done = True

        return s_plus, r, self.done

    def render(self):
        pass

    def select_action(self):
        pass

    @staticmethod
    def resize(data, scale_factor, size_divisor=32):
        # resize image according to scale factor.
        img = data['img'][0]
        img_meta = data['img_meta'][0]

        img = nn.functional.interpolate(img, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        device = img.device
        img_np = np.transpose(img.squeeze().numpy(), (1, 2, 0))

        # The size of last feature map is 1/32(specially for resnet50) of the original image,
        # Impad to ensure that the original image size is multiple of 32.
        img_np = mmcv.impad_to_multiple(img_np, size_divisor)
        data['img'][0]= torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)

        # change image meta info according to scale factor change.
        img_meta[0][0]['img_shape'] = (int(img_meta[0][0]['img_shape'][0] * scale_factor),
                                    int(img_meta[0][0]['img_shape'][1] * scale_factor), 3)
        img_meta[0][0]['pad_shape'] = img_np.shape
        img_meta[0][0]['scale_factor'] = img_meta[0][0]['scale_factor'] * scale_factor

        data['img'][0] = img
        data['img_meta'][0] = img_meta

        return data
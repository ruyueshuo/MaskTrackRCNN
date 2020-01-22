#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/19 上午9:50
# @Author  : FengDa
# @File    : utils.py
# @Software: PyCharm
import numpy as np
import torchvision.models as models


def get_self_feat(input):
    '''
    :param self:
    :param img_tensor: a tensor with size of batchsize * channels * height * weight
    :return:
    '''
    model = models.resnet50(pretrained = True)
    model.eval()

    x=input
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    # print(x.size())

    return x

def get_diff_feat(self, full_data, current_data):
    full_data = self.resize(full_data, scale_factor=self.actions[-1])
    return full_data['img'] - current_data['img']


class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X
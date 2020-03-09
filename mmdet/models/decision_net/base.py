#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/2 14:04
# @Author  : FengDa
# @File    : base.py
# @Software: PyCharm

import logging
from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset
import os
import cv2
import mmcv
import numpy as np
import torch.nn as nn


class BaseRL(nn.Module):
    """Base class for RL algorithm."""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseRL, self).__init__()

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def store_transition(self, *args):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def save_model(self, save_path):
        pass


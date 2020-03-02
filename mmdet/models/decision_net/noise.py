#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/29 17:57
# @Author  : FengDa
# @File    : noise.py
# @Software: PyCharm
import numpy as np

class OrnsteinUhlenbeckActionNoise:
    """Add Ornstein-Uhlenbeck noise to action."""
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
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


class GaussianActionNoise(object):
    """Add gaussian noise to action."""
    def __init__(self, action_dim, mu=0, sigma=0.5):
        self.action_dim = action_dim
        self.mu = mu
        self.sigma = sigma

    def reset(self):
        pass

    def sample(self):
        dx = np.random.normal(self.mu, self.sigma, self.action_dim)
        return dx
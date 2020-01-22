#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/14 下午1:39
# @Author  : FengDa
# @File    : memory.py
# @Software: PyCharm
import random
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))
        # return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)


class ReplayBuffer(object):

    def __init__(self, capacity, seed,
                 priority_weight=None, priority_exponent=None,
                 priotirized_experience=False):
        self.capacity = capacity
        self.position = 0
        self.prioritize = priotirized_experience
        self.priority_weight = priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = priority_exponent
        if self.prioritize:
            self.memory = []
        else:
            self.memory = []
        # Seed for reproducible results
        np.random.seed(seed)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_buffer_size(self):
        return len(self.memory)
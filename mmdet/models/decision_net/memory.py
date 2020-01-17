#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/14 下午1:39
# @Author  : FengDa
# @File    : memory.py
# @Software: PyCharm
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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
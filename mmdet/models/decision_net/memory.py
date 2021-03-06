#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/14 下午1:39
# @Author  : FengDa
# @File    : memory.py
# @Software: PyCharm
import os
import random
# import pickle
import _pickle as pickle
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
                 priotirized_experience=False,
                 resume=False,
                 save_path="/home/ubuntu/data/replaybuffertest10w/"):
        self.capacity = capacity
        self.position = 0
        self.prioritize = priotirized_experience
        self.priority_weight = priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = priority_exponent
        if self.prioritize:
            self.memory = []
        else:
            self.memory = []

        # Resuming from an existing replay buffer from hard disk.
        if resume:
            self.memory = [i for i in range(self.capacity)]

        # Seed for reproducible results
        np.random.seed(seed)
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def push_hd(self, *args):
        """Save a transition to hard disk."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.position
        file_name = os.path.join(self.save_path, "{}.pkl".format(self.position))
        save_trans(Transition(*args), file_name)
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_batch_hd(self, batch_size):
        idxes = random.sample(self.memory, batch_size)
        # from multiprocessing.pool import Pool
        # dataset = [os.path.join(self.save_path, "{}.pkl".format(idx)) for idx in idxes]
        # p = Pool(4)
        # transitions = p.map(load_trans, dataset)
        # for i in range(10):
        #     # 创建进程，放入进程池统一管理
        #     p.apply_async(load_trans, args=(i,))
        # import concurrent.futures
        # dataset = [os.path.join(self.save_path, "{}.pkl".format(idx)) for idx in idxes]
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #
        #     transitions = executor.map(load_trans, dataset)
        # t = transitions.result()
        transitions = [load_trans(os.path.join(self.save_path, "{}.pkl".format(idx))) for idx in idxes]
        return transitions

    def __len__(self):
        return len(self.memory)

    def get_buffer_size(self):
        return len(self.memory)


class PrioritizedReplayBuffer(object):
    """Prioritized Replay Buffer."""
    def __init__(self, capacity,
                 seed=41,
                 resume=False,
                 save_path="/home/ubuntu/data/replaybuffertest10w/"):
        self.capacity = capacity
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.tree = SumTree(capacity)
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority, default is 0.6.
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1, default is 0.4.
        self.beta_increment_per_sampling = 0.00001  # default is 0.001.
        self.abs_err_upper = 1.  # clipped abs error, default is 1..
        self.memory = []
        # Resuming from an existing replay buffer from hard disk.
        if resume:
            self.memory = [i for i in range(self.capacity)]

        self.position = 0
        np.random.seed(seed)

    def push_hd(self, *args):
        """Save a transition to hard disk."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.position
        file_name = os.path.join(self.save_path, "{}.pkl".format(self.position))
        save_trans(Transition(*args), file_name)
        self.position = (self.position + 1) % self.capacity

        self.store(np.array(self.position))

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))

        pri_seg = self.tree.total_p / n  # priority segment

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def sample_batch_hd(self, batch_size):
        idxes, memories, ISWeights = self.sample(batch_size)
        # idxes = random.sample(self.memory, batch_size)
        data_idxes = [_idx - self.capacity + 1 for _idx in idxes]
        import concurrent.futures
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     dataset = [os.path.join(self.save_path, "{}.pkl".format(idx)) for idx in data_idxes]
        #     transitions = executor.map(load_trans, dataset)
        transitions = [load_trans(os.path.join(self.save_path, "{}.pkl".format(idx))) for idx in data_idxes]
        return idxes, transitions, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def get_buffer_size(self):
        return len(self.memory)


class SumTree(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.data_pointer = 0
        # Build tree.
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    @property
    def total_p(self):
        return self.tree[0]  # the root

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        # 在添加数据的时，由于某个叶子节点的优先级数值变化，那么它一系列父节点的数值也会发生变化，用update更新
        # 当 sample 被 train, 有了新的 TD-error, 就在 tree 中更新
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
         Tree structure and array storage:
         Tree index:
              0         -> storing priority sum
             / \
           1     2
          / \   / \
         3   4 5   6    -> storing priority for transitions
         Array type for storing:
         [0,1,2,3,4,5,6]
         """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1  # # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


def save_trans(trans, filename):
    """
    Save feature tensor to hard disk.
    :param trans: transition to be saved.
    :param filename: file name.
    """
    # state = [s.clone().cpu().detach().numpy() for s in trans.state]
    # next_state = [s.clone().cpu().detach().numpy() for s in trans.next_state]
    # # action = trans.action.clone().cpu().detach().numpy()
    # # reward = trans.reward.clone().cpu().detach().numpy()
    # # done = trans.done.clone().cpu().detach().numpy()
    # action = trans.action
    # reward = trans.reward
    # done = trans.done
    # trans = (state, action, next_state, reward, done)
    with open(filename, 'wb') as f:
        pickle.dump(trans, f)

    return True


def load_trans(file):
    with open(file, 'rb') as f:
        trans = pickle.load(f)
    return trans



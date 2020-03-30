#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/14 下午1:38
# @Author  : FengDa
# @File    : dqn.py
# @Software: PyCharm
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from .memory import ReplayBuffer, Transition, PrioritizedReplayBuffer
from .ddpg import ActorNet as Net
from .base import BaseRL


class DQN(BaseRL):
    """docstring for DQN"""

    def __init__(self,
                 input_channel,
                 num_outputs,
                 buffer_capacity=10000,
                 random_seed=41,
                 lr=0.0001,
                 gamma=0.99,
                 batch_size=32,
                 episilo=0.5,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=500000,
                 priority=True,
                 criterion=nn.MSELoss(),
                 checkpoint=None,
                 DDQN=True):
        super(DQN, self).__init__()
        self.input_channel = input_channel
        self.num_outputs = num_outputs
        self.gamma = gamma
        self.batch_size = batch_size
        self.episilo = episilo
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.priority = priority
        self.DDQN = DDQN

        self.policy_net, self.target_net \
            = Net(input_channel, num_outputs, checkpoint=checkpoint), Net(input_channel, num_outputs, checkpoint=checkpoint)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.learn_step_counter = 0
        self.steps_done = 0

        if self.priority:
            self.buffer = PrioritizedReplayBuffer(capacity=buffer_capacity, seed=random_seed, resume=False)
        else:
            self.buffer = ReplayBuffer(capacity=buffer_capacity, seed=random_seed, resume=True)

        # Todo learning rate decay(high to low)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_func = criterion

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            self.to_cuda()

    def name(self):
        return "DQN"

    def to_cuda(self):
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)

    def select_action(self, state):
        # Select action by policy net or randomly with certain probability.
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return np.argmax(np.random.rand(self.num_outputs))
            # return np.array([random.randrange(self.num_outputs)])

    def choose_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def store_transition(self, state, new_state, action, reward, done):
        # Store the transition into the replay buffer
        self.buffer.push_hd(state, action, new_state, reward, done)

    def learn(self, times=5):
        # Update the parameters
        Q_NETWORK_ITERATION = 50
        for _ in range(times):
            if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            self.learn_step_counter += 1

            # Sample batch from memory
            if self.priority:
                tree_idx, transitions, ISWeights = self.buffer.sample_batch_hd(self.batch_size)
            else:
                transitions = self.buffer.sample_batch_hd(self.batch_size)
            batch = Transition(*zip(*transitions))

            def batch_state(batch):
                states = []
                states.append(torch.cat([state[0] for state in batch.state]).cuda())
                states.append(torch.cat([state[1] for state in batch.state]).cuda())
                states.append(torch.cat([state[2].float().unsqueeze(dim=0) for state in batch.state]).cuda())
                states.append(torch.cat([state[3].unsqueeze(dim=0) for state in batch.state]).cuda())

                new_states = []
                new_states.append(torch.cat([state[0] for state in batch.next_state]).cuda())
                new_states.append(torch.cat([state[1] for state in batch.next_state]).cuda())
                new_states.append(torch.cat([state[2].float().unsqueeze(dim=0) for state in batch.next_state]).cuda())
                new_states.append(torch.cat([state[3].unsqueeze(dim=0) for state in batch.next_state]).cuda())

                actions = torch.cat([torch.tensor(action).cuda().float().reshape((-1, 1)) for action in batch.action])

                rewards = torch.cat([torch.tensor(reward).cuda().float().reshape((-1, 1)) for reward in batch.reward])
                dones = torch.cat([torch.tensor(done).cuda().float().reshape((-1, 1)) for done in batch.done])

                return states, new_states, actions, rewards, dones

            # Get the separate values from the named tuple
            states, new_states, actions, rewards, dones = batch_state(batch)

            # Compute  loss
            q_eval = self.policy_net(states).gather(1, actions.type(torch.LongTensor).to(self.device))
            if self.DDQN:
                with torch.no_grad():
                    eval_next_act_batch = self.policy_net(new_states).max(1)[1][:, None]
                    target_next_val_batch = self.target_net(new_states).gather(1, eval_next_act_batch)
                    q_target = rewards + self.gamma * target_next_val_batch
            else:
                with torch.no_grad():
                    q_next = self.target_net(new_states).detach()
                    q_target = rewards + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

            if self.priority:
                # Loss:mse, losses = sum(weight*loss)
                ISWeights = torch.from_numpy(ISWeights).cuda().float().sqrt()
                loss = self.loss_func(ISWeights*q_eval, ISWeights*q_target)
            else:
                loss = self.loss_func(q_eval, q_target)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1.1, 1.1)
            self.optimizer.step()

            # Update priority in SumTree
            if self.priority:
                td_errors = (q_target - q_eval).cpu().detach().numpy()
                self.buffer.batch_update(tree_idx=tree_idx, abs_errors=np.abs(td_errors))

    def save_model(self, output):
        """Saving model."""
        print("Saving model.")
        torch.save(
            self.policy_net.state_dict(),
            '{}/policy_net.pkl'.format(output)
        )
        return True

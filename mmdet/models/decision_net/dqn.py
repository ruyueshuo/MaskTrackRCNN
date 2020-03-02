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
from .memory import ReplayBuffer, Transition
from .ddpg import ActorNet as Net
from .base import BaseRL
# class Net(nn.Module):
#
#     def __init__(self,
#                  input_channel,
#                  num_outputs,
#                  checkpoint=None,
#                  is_training=True ):
#         super(Net, self).__init__()
#
#         self.conv1 = nn.Conv2d(input_channel, 512, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(512)
#         # self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False)
#         # self.bn2 = nn.BatchNorm2d(64)
#
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#
#         self.fc1 = nn.Linear(7680*4, 1024*2)
#         self.fc2 = nn.Linear(1024*2, 256)
#         self.fc3 = nn.Linear(256, 64)
#         self.fc4 = nn.Linear(64+11, num_outputs)
#
#         # Activation functions
#         self.relu = nn.ReLU(inplace=True)
#         self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#         self.activation = self.leakyrelu
#
#         self.is_training = is_training
#         self.checkpoint = checkpoint
#         self.init_weights()
#
#     def init_weights(self):
#         # super(ActorNet, self).init_weights()
#         if self.checkpoint:
#             """load checkpoint."""
#             load_checkpoint(self, self.checkpoint)
#         else:
#             for m in self.children():
#                 if isinstance(m, nn.Conv2d):
#                     if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
#                         nn.init.kaiming_normal_(m.weight, mode='fan_in')
#                     elif isinstance(self.activation, nn.Sigmoid):
#                         nn.init.xavier_uniform_(m.weight)
#                     else:
#                         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                         m.weight.data.normal_(0, math.sqrt(2. / n))
#
#                 elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#                     m.weight.data.fill_(1)
#                     m.bias.data.zero_()
#
#                 elif isinstance(m, nn.Linear):
#                     nn.init.kaiming_normal_(m.weight)
#                     # nn.init.xavier_uniform_(m.weight)
#
#     def forward(self, inputs):
#         """mu, sigma_sq?"""
#         feat_FAR = inputs[2].clone().detach().reshape((-1, 1))
#         feat_history = inputs[3].reshape(-1, 10)
#         x = torch.cat((inputs[0], inputs[1]), 1)
#         y = torch.cat((feat_FAR, feat_history), 1)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         # if self.is_training:
#         #     x = self.bn1(x)
#         x = self.activation(x)
#         # x = self.conv2(x)
#         # if self.is_training:
#         #     x = self.bn2(x)
#         # x = self.relu(x)
#         x = self.maxpool(x)
#         x = x.view(x.size(0), -1)
#         try:
#             x = self.activation(self.fc1(x))
#         except RuntimeError as e:
#             print("x.shape:{}\tfeat_self.shape:{}".format(x.shape, inputs[1].shape))
#         x = self.activation(self.fc2(x))
#         x = self.activation(self.fc3(x))
#         x = torch.cat((x, y), 1)
#         x = self.fc4(x)
#         x = self.sigmoid(x)
#         # x = self.softmax(x)
#
#         return x


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
                 eps_decay=20000):
        super(DQN, self).__init__()
        self.input_channel = input_channel
        self.num_outputs = num_outputs
        self.gamma = gamma
        self.batch_size = batch_size
        self.episilo = episilo
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.policy_net, self.target_net = Net(input_channel, num_outputs), Net(input_channel, num_outputs)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.learn_step_counter = 0
        self.steps_done = 0
        self.buffer = ReplayBuffer(capacity=buffer_capacity, seed=random_seed)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            self.to_cuda()

    def name(self):
        return "DQN"

    # def choose_action(self, state):
    #     # state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
    #     if np.random.randn() <= self.episilo:# greedy policy
    #         action_value = self.policy_net.forward(state)
    #         action = torch.max(action_value, 1)[1].data.numpy()
    #         action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
    #     else: # random policy
    #         action = np.random.randint(0, NUM_ACTIONS)
    #         action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
    #     return action
    def to_cuda(self):
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)

    def select_action(self, state):
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

    # Store the transition into the replay buffer
    def store_transition(self, state, new_state, action, reward, done):
        self.buffer.push_hd(state, action, new_state, reward, done)

    def learn(self):
        #update the parameters
        Q_NETWORK_ITERATION = 10
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        transitions = self.buffer.sample_batch_hd(self.batch_size)
        batch = Transition(*zip(*transitions))

        def batch_state(batch):
            states = []
            states.append(torch.cat([state[0] for state in batch.state]))
            states.append(torch.cat([state[1] for state in batch.state]))
            states.append(torch.cat([state[2].float().unsqueeze(dim=0) for state in batch.state]))
            states.append(torch.cat([state[3].unsqueeze(dim=0) for state in batch.state]))

            new_states = []
            new_states.append(torch.cat([state[0] for state in batch.next_state]))
            new_states.append(torch.cat([state[1] for state in batch.next_state]))
            new_states.append(torch.cat([state[2].float().unsqueeze(dim=0) for state in batch.next_state]))
            new_states.append(torch.cat([state[3].unsqueeze(dim=0) for state in batch.next_state]))

            actions = torch.cat([torch.tensor(action).cuda().float().reshape((-1, 1)) for action in batch.action])

            rewards = torch.cat([torch.tensor(reward).cuda().float().reshape((-1, 1)) for reward in batch.reward])
            dones = torch.cat([torch.tensor(done).cuda().float().reshape((-1, 1)) for done in batch.done])

            return states, new_states, actions, rewards, dones

        # Get the separate values from the named tuple
        states, new_states, actions, rewards, dones = batch_state(batch)

        # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # batch_memory = self.memory[sample_index, :]
        # batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        # batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        # batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        # batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval
        q_eval = self.policy_net(states).gather(1, actions.type(torch.LongTensor).to(self.device))
        q_next = self.target_net(new_states).detach()
        q_target = rewards + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, output):
        """
        Saving the models
        :param output:
        :return:
        """
        print("Saving model.")
        torch.save(
            self.policy_net.state_dict(),
            '{}/policy_net.pkl'.format(output)
        )
        # torch.save(
        #     self.critic.state_dict(),
        #     '{}/critic.pkl'.format(output)
        # )
        return True

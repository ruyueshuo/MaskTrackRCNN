#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/16 上午10:39
# @Author  : FengDa
# @File    : policy_gradient.py
# @Software: PyCharm
import math
import numpy as np
import torch
import torch.nn as nn

from mmcv.runner import load_checkpoint

pi = torch.FloatTensor([math.pi]).cuda()


class PolicyNet(nn.Module):
    """Policy Network."""

    def __init__(self,
                 input_channel,
                 num_outputs,
                 hidden_size=100,
                 checkpoint=None):
        """
        :param input_size:
        :param hidden_size:
        :param action_space:
        """
        super(PolicyNet, self).__init__()
        # self.action_space = action_space
        # num_outputs = action_space.shape[0]

        self.conv = nn.Conv2d(input_channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc1 = nn.Linear(3840, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128+6, num_outputs)

        self.sigmoid = nn.Sigmoid()

        self.checkpoint = checkpoint

    def init_weights(self):
        super(PolicyNet, self).init_weights()
        if self.checkpoint:
            """load checkpoint."""
            load_checkpoint(self, self.checkpoint)
        else:
            for _module in [self.conv, self.fc1, self.fc2, self.fc3, self.fc4]:
                nn.init.xavier_uniform_(_module.weight)
                nn.init.constant_(_module.bias, 0)

    def forward(self, inputs):
        """mu, sigma_sq?"""
        feat_diff = inputs[0].cuda()
        feat_self = inputs[1].cuda()
        feat_FAR = torch.tensor(inputs[2]).cuda().float().reshape((-1, 1))
        feat_history = inputs[3].reshape(-1, 5)
        x = torch.cat((feat_diff, feat_self), 1)
        y = torch.cat((feat_FAR, feat_history), 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.cat((x, y), 1)
        x = self.sigmoid(self.fc4(x))
        # x = self.softmax(x)
        # x = self.conv(x)
        # x = nn.Relu(self.linear1(x))
        # mu = self.linear2(x)
        # sigma_sq = self.linear2_(x)

        # return mu, sigma_sq
        return x


class PolicyGradient:
    """Policy Gradient Algorithm."""

    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.95,
                 output_graph=True):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # reward 递减率

        self.states, self.actions, self.rewards = [], [], []  # 这是我们存储 回合信息的 list

        self.model = PolicyNet(n_features, 100, n_actions)
        #
        # if output_graph:  # 是否输出 tensorboard 文件
        #     # $ tensorboard --logdir=logs
        #     # http://0.0.0.0:6006/
        #     # tf.train.SummaryWriter soon be deprecated, use following
        #     tf.summary.FileWriter(r'D:\logs', self.sess.graph)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})    # 所有 action 的概率
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # 根据概率来选 action
        return action

    def select_action(self, state):
        mu, sigma_sq = self.model(Variable(state).cuda())
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())
        # calculate the probability
        action = (mu + sigma_sq.sqrt()*Variable(eps).cuda()).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1)

        log_prob = prob.log()
        return action, log_prob, entropy


    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # 衰减, 并标准化这回合的 reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()  # 功能再面

        # train on episode
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # 清空回合 data
        return discounted_ep_rs_norm  # 返回这一回合的 state-action value

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


class REINFORCE(object):

    def __init__(self, num_inputs, num_outputs):
        # self.action_space = action_space
        self.model = PolicyNet(num_inputs, num_outputs)
        self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def select_action(self, state):
        a_prob = self.model(state)
        a = torch.argmax(a_prob)
        return a, a_prob

    # def select_action(self, state):
    #     mu, sigma_sq = self.model(state.cuda())
    #     sigma_sq = nn.functional.softplus(sigma_sq)
    #
    #     eps = torch.randn(mu.size())
    #     # calculate the probability
    #     action = (mu + sigma_sq.sqrt() * (eps).cuda()).data
    #     prob = self.normal(action, mu, sigma_sq)
    #     entropy = -0.5 * ((sigma_sq + 2 * pi.expand_as(sigma_sq)).log() + 1)
    #
    #     log_prob = prob.log()
    #     return action, log_prob, entropy
    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            # loss = loss - (log_probs[i] * ((R).expand_as(log_probs[i])).cuda()).sum() - (
            #             0.0001 * entropies[i].cuda()).sum()
        loss = R
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()

    @staticmethod
    def normal(x, mu, sigma_sq):
        a = (-1 * ((x) - mu).pow(2) / (2 * sigma_sq)).exp()
        b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
        return a * b

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
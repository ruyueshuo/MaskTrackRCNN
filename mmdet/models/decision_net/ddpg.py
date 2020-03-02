#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/20 下午2:18
# @Author  : FengDa
# @File    : ddpg.py
# @Software: PyCharm

# This script contains the Actor and Critic classes
import math
import torch
import numpy as np

import torch.optim as opt
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .noise import OrnsteinUhlenbeckActionNoise, GaussianActionNoise
from .memory import ReplayBuffer, Transition

from mmcv.runner import load_checkpoint


class DDPG(object):
    """
    The Deep Deterministic policy gradient network
    """

    def __init__(self, num_hidden_units=128, input_dim=512, num_actions=1, num_q_val=1,
                 observation_dim=None, goal_dim=None,
                 batch_size=8, use_cuda=False, gamma=0.99, random_seed=41,
                 actor_optimizer=opt.Adam, critic_optimizer=opt,
                 actor_learning_rate=0.001, critic_learning_rate=0.001,
                 loss_function=F.smooth_l1_loss, polyak_constant=0.05,
                 buffer_capacity=1000, non_conv=False,
                 num_conv_layers=None, num_pool_layers=None,
                 conv_kernel_size=None, img_height=None, img_width=None,
                 input_channels=None, a_check_point=None, c_check_point=None):

        self.num_hidden_units = num_hidden_units
        self.non_conv = non_conv
        self.num_actions = num_actions
        self.num_q = num_q_val
        self.obs_dim = observation_dim
        self.goal_dim = goal_dim
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.cuda = use_cuda
        self.gamma = gamma
        self.seed(random_seed)
        self.actor_optim = actor_optimizer
        self.critic_optim = critic_optimizer
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.criterion = loss_function
        self.tau = polyak_constant
        self.buffer = ReplayBuffer(capacity=buffer_capacity, seed=random_seed)
        self.a_check_point = a_check_point
        self.c_check_point = c_check_point

        # Convolution Parameters
        self.num_conv = num_conv_layers
        self.pool = num_pool_layers
        self.im_height = img_height
        self.im_width = img_width
        self.conv_kernel_size = conv_kernel_size
        self.input_channels = input_channels

        # Actor & Critic Net Initialization.
        self.target_actor = ActorNet(input_dim, num_actions, a_check_point)
        self.actor = ActorNet(input_dim, num_actions, a_check_point)
        self.target_critic = CriticNet(input_dim, num_q_val, c_check_point)
        self.critic = CriticNet(input_dim, num_q_val, c_check_point)

        if self.cuda:
            self.to_cuda()

        # Initializing the target networks with the standard network weights
        # self.target_actor.load_state_dict(self.actor.state_dict())
        # self.target_critic.load_state_dict(self.critic.state_dict())

        # Create the optimizers for the actor and critic using the corresponding learning rate
        actor_parameters = self.actor.parameters()
        critic_parameters = self.critic.parameters()

        self.actor_optim = opt.Adam(actor_parameters, lr=self.actor_lr)
        self.critic_optim = opt.Adam(critic_parameters, lr=self.critic_lr)

        # Initialize a random exploration noise
        # self.random_noise = ](self.num_actions)
        self.random_noise = GaussianActionNoise(self.num_actions)

        self.idx = 0

    def to_cuda(self):
        self.target_actor = self.target_actor.cuda()
        self.target_critic = self.target_critic.cuda()
        self.actor = self.actor.cuda()
        self.critic = self.critic.cuda()

    def save_model(self, output):
        """
        Saving the models
        :param output:
        :return:
        """
        print("Saving the actor and critic")
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def random_action(self):
        """
        Take a random action bounded between min and max values of the action space
        :return:
        """
        action = np.random.uniform(-1., 1., self.num_actions)
        self.a_t = action

        return action

    def seed(self, s):
        """
        Setting the random seed for a particular training iteration
        :param s:
        :return:
        """
        torch.manual_seed(s)
        if self.cuda:
            torch.cuda.manual_seed(s)

    def get_actors(self):
        return {'target': self.target_actor, 'actor': self.actor}

    def get_critics(self):
        return {'target': self.target_critic, 'critic': self.critic}

    # Get the action with an option for random exploration noise
    def get_action(self, state, noise=True):
        # state_v = Variable(state)
        action = self.actor(state)
        if noise:
            noise = self.random_noise
            action = action.data.cpu().numpy()[0] + noise.sample()
        else:
            action = action.data.cpu().numpy()[0]
        action = np.clip(action, 0, 1.)
        return action

    # Reset the noise
    def reset(self):
        self.random_noise.reset()

    # Store the transition into the replay buffer
    def store_transition(self, state, new_state, action, reward, done):
        self.buffer.push_hd(state, action, new_state, reward, done)

    # Update the target networks using polyak averaging
    def update_target_networks(self):
        self.idx += 1
        if self.idx % 5 == 0:
            self.idx = 0
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    # Calculate the Temporal Difference Error
    def calc_td_error(self, transition):
        """
        Calculates the td error against the bellman target
        :return:
        """
        # Calculate the TD error only for the particular transition

        # Get the separate values from the named tuple
        state, new_state, reward, success, action, done = transition

        state = Variable(state)
        new_state = Variable(new_state)
        reward = Variable(reward)
        action = Variable(action)
        done = Variable(done)

        if self.cuda:
            state = state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            new_state = new_state.cuda()
            done = done.cuda()

        new_action = self.target_actor(new_state)
        next_Q_value = self.target_critic(new_state, new_action)
        # Find the Q-value for the action according to the target actior network
        # We do this because calculating max over a continuous action space is intractable
        next_Q_value.volatile = False
        next_Q_value = torch.squeeze(next_Q_value, dim=1)
        next_Q_value = next_Q_value * (1 - done)
        y = reward + self.gamma * next_Q_value

        outputs = self.critic(state, action)
        td_loss = self.criterion(outputs, y)
        return td_loss

    # Train the networks
    def fit_batch(self):
        # Sample mini-batch from the buffer uniformly or using prioritized experience replay

        # If the size of the buffer is less than batch size then return
        if self.buffer.get_buffer_size() < self.batch_size:
            return None, None

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

        # Step 2: Compute the target values using the target actor network and target critic network
        # Compute the Q-values given the current state ( in this case it is the new_states)
        #with torch.no_grad():

        new_action = self.target_actor(new_states)
        # new_action.volatile = True
        with torch.no_grad():
            next_Q_values = self.target_critic(new_states, new_action)
        # Find the Q-value for the action according to the target actior network
        # We do this because calculating max over a continuous action space is intractable
        # next_Q_values.volatile = False
        # next_Q_values = torch.squeeze(next_Q_values, dim=1)
        # dones = torch.squeeze(dones, dim=1)
        next_Q_values = next_Q_values * (1 - dones)
        # next_Q_values.volatile = False
        with torch.no_grad():
            y = rewards + self.gamma*next_Q_values

        # Zero the optimizer gradients
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        # Forward pass
        outputs = self.critic(states, actions)
        loss = self.criterion(outputs, y)
        loss.backward(retain_graph=True)
        # Clamp the gradients to avoid vanishing gradient problem
        for param in self.critic.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.critic_optim.step()

        # Updating the actor policy
        policy_loss = -1 * self.critic(states, self.actor(states))  # TODO 没搞明白？Q的负数作为loss?
        policy_loss = policy_loss.mean()
        policy_loss.backward(retain_graph=True)
        # Clamp the gradients to avoid the vanishing gradient problem
        for param in self.actor.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.actor_optim.step()

        return loss, policy_loss


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ActorNet(nn.Module):
    """Actor Network. Input state, output action.
    """

    def __init__(self,
                 input_channel,
                 num_outputs,
                 checkpoint=None,
                 is_training=True):

        super(ActorNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        # self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc1 = nn.Linear(7680*4, 1024*2)
        self.fc2 = nn.Linear(1024*2, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64+11, num_outputs)

        # self.bn_fc1 = nn.BatchNorm1d(1024*2)
        # self.bn_fc2 = nn.BatchNorm1d(256)
        # self.bn_fc3 = nn.BatchNorm1d(64)

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.activation = self.leakyrelu

        self.is_training = is_training
        self.checkpoint = checkpoint
        self.init_weights()

    def init_weights(self):
        # super(ActorNet, self).init_weights()
        if self.checkpoint:
            """load checkpoint."""
            load_checkpoint(self, self.checkpoint)
        else:
            for m in self.children():
                if isinstance(m, nn.Conv2d):
                    if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                        nn.init.kaiming_normal_(m.weight, mode='fan_in')
                    elif isinstance(self.activation, nn.Sigmoid):
                        nn.init.xavier_uniform_(m.weight)
                    else:
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))

                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    # nn.init.xavier_uniform_(m.weight)

            # for _module in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3, self.fc4]:
            #     if self.activation in [self.relu, self.leakyrelu]:
            #         nn.init.kaiming_normal_(_module.weight, mode='fan_in')
            #     else:
            #         nn.init.xavier_uniform_(_module.weight)
            #
            #     elif isinstance(_module, nn.BatchNorm2d):
            #         _module.weight.data.fill_(1)
            #         _module.bias.data.zero_()
                # if _module not in [self.conv1, self.conv2]:
                #     # _module.bias.fill_(0.01)
                #     nn.init.constant_(_module.bias, 0)

    def forward(self, inputs):
        """mu, sigma_sq?"""
        # feat_diff = inputs[0]
        # feat_self = inputs[1]
        feat_FAR = inputs[2].clone().detach().reshape((-1, 1))
        feat_history = inputs[3].reshape(-1, 10)
        x = torch.cat((inputs[0], inputs[1]), 1)
        y = torch.cat((feat_FAR, feat_history), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        # if self.is_training:
        #     x = self.bn1(x)
        x = self.activation(x)
        # x = self.conv2(x)
        # if self.is_training:
        #     x = self.bn2(x)
        # x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        try:
            x = self.activation(self.fc1(x))
        except RuntimeError as e:
            print("x.shape:{}\tfeat_self.shape:{}".format(x.shape, inputs[1].shape))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = torch.cat((x, y), 1)
        x = self.fc4(x)
        x = self.sigmoid(x)
        # x = self.softmax(x)

        return x


class CriticNet(nn.Module):
    """
    CriticNet Network. Input state and action, output q."""

    def __init__(self,
                 input_channel,
                 num_outputs,
                 checkpoint=None,
                 is_training=False):

        super(CriticNet, self).__init__()

        self.conv = nn.Conv2d(input_channel, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc1 = nn.Linear(7680, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64+15, num_outputs)

        self.bn_fc1 = nn.BatchNorm1d(1024)
        # self.bn_fc2 = nn.BatchNorm1d(256)
        # self.bn_fc3 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.activation = self.relu
        self.is_training = is_training
        self.checkpoint = checkpoint
        self.init_weights()

    def init_weights(self):
        # super(CriticNet, self).init_weights()
        if self.checkpoint:
            """load checkpoint."""
            load_checkpoint(self, self.checkpoint)
        else:
            for m in self.children():
                if isinstance(m, nn.Conv2d):
                    if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                        nn.init.kaiming_normal_(m.weight, mode='fan_in')
                    elif isinstance(self.activation, nn.Sigmoid):
                        nn.init.xavier_uniform_(m.weight)
                    else:
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))

                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
            # for _module in [self.conv, self.fc1, self.fc2, self.fc3, self.fc4]:
            #     nn.init.xavier_uniform_(_module.weight)
            #     if _module not in [self.conv]:
            #         # _module.bias.fill_(0.01)
            #         nn.init.constant_(_module.bias, 0)

    def forward(self, state, action):
        """mu, sigma_sq?"""
        # feat_diff = state[0]
        # feat_self = state[1]
        feat_FAR = state[2].clone().detach().float().reshape((-1, 1))
        feat_history = state[3].reshape(-1, 10)
        x = torch.cat((state[0], state[1]), 1)
        y = torch.cat((feat_FAR, feat_history), 1)
        x = self.conv(x)
        if self.is_training:
            x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        # add action
        x = torch.cat((x, y, action), 1)
        x = self.fc4(x)
        # x = self.softmax(x)

        return x


class ActorDDPGNetwork(nn.Module):
    # The actor network takes the state as input and outputs an action
    # The actor network is used to approximate the argmax action in a continous action space
    # The actor network in the case of a discrete action space is just argmax_a(Q(s,a))

    def __init__(self, num_conv_layers, conv_kernel_size, input_channels, output_action, dense_layer,
                 pool_kernel_size, IMG_HEIGHT, IMG_WIDTH):
        super(ActorDDPGNetwork, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.conv_kernel = conv_kernel_size
        self.input_channels = input_channels
        self.output_action = output_action
        self.dense_layer = dense_layer
        self.pool_kernel_size = pool_kernel_size
        self.im_height = IMG_HEIGHT
        self.im_width = IMG_WIDTH

        # Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_conv_layers, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=num_conv_layers, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.kernel_size)
        self.bn3 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu3 = nn.ReLU(inplace=True)

        # Fully connected layer
        self.fully_connected_layer = nn.Linear(234432, self.dense_layer)
        self.relu4 = nn.ReLU(inplace=True)
        self.output_layer = nn.Linear(self.dense_layer, output_action)

        # Weight initialization from a uniform gaussian distribution
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer(x)
        x = self.relu4(x)
        out = self.output_layer(x)
        return out


# For non image state space
class ActorDDPGNonConvNetwork(nn.Module):
    def __init__(self, num_hidden_layers, output_action, input):
        super(ActorDDPGNonConvNetwork, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.input = input
        self.output_action = output_action
        self.init_w = 3e-3

        #Dense Block
        self.dense_1 = nn.Linear(self.input, self.num_hidden_layers)
        self.relu1 = nn.ReLU(inplace=True)
        self.dense_2 = nn.Linear(self.num_hidden_layers, self.num_hidden_layers)
        self.relu2 = nn.ReLU(inplace=True)
        self.output = nn.Linear(self.num_hidden_layers, self.output_action)
        self.tanh = nn.Tanh()

    def init_weights(self, init_w):
        self.dense_1.weight.data = fanin_init(self.dense_1.weight.data.size())
        self.dense_2.weight.data = fanin_init(self.dense_2.weight.data.size())
        self.output.weight.data.uniform_(-init_w, init_w)

    def forward(self, input):
        x = self.dense_1(input)
        x = self.relu1(x)
        x = self.dense_2(x)
        x = self.relu2(x)
        output = self.output(x)
        output = self.tanh(output)
        return output


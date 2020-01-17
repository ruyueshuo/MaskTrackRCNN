#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/17 下午12:33
# @Author  : FengDa
# @File    : train_rl.py
# @Software: PyCharm

import argparse, math, os
import numpy as np
# import gym
# from gym import wrappers

import torch
from torch.autograd import Variable
import torch.nn.utils as utils
import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import scatter, collate, MMDataParallel
from mmdet.models import build_detector, detectors
# from normalized_actions import NormalizedActions
from mmdet.models.decision_net.env import DecisionEnv
from mmdet.models.decision_net.policy_gradient import REINFORCE

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
# Test arguments
parser.add_argument('--env_name', type=str, default='CartPole-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=2000, metavar='N',
                    help='number of episodes (default: 2000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--ckpt_freq', type=int, default=100,
                    help='model saving frequency')
parser.add_argument('--display', type=bool, default=False,
                    help='display or not')

# MaskTrackRCNN arguments
parser.add_argument('--config', default="../configs/masktrack_rcnn_r50_fpn_1x_flow_youtubevos.py",
                    help='test config file path')
parser.add_argument('--checkpoint', default="../results/baseline/epoch_12.pth", help='checkpoint file')
parser.add_argument('--checkpointflow', default="../pretrained_models/flownetc_EPE1.766.tar", help='checkpoint file')
parser.add_argument(
    '--save_path', default="/home/ubuntu/datasets/YT-VIS/results/",
    type=str,
    help='path to save visual result')
parser.add_argument(
    '--gpus', default=1, type=int, help='GPU number used for testing')
parser.add_argument(
    '--proc_per_gpu',
    default=1,
    type=int,
    help='Number of processes per GPU')
parser.add_argument('--out', help='output result file')
parser.add_argument('--load_result',
                    default=False,
                    # action='store_true',
                    help='whether to load existing result')
parser.add_argument(
    '--eval',
    default=['bbox', 'segm'],
    type=str,
    nargs='+',
    choices=['bbox', 'segm'],
    help='eval types')
# parser.add_argument('--show', action='store_true', help='show results')
parser.add_argument('--show', default=True, help='show results')

args = parser.parse_args()

# config
cfg = mmcv.Config.fromfile(args.config)

# build task model
assert args.gpus == 1
model = build_detector(
    cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
load_checkpoint(model, args.checkpoint)
model = MMDataParallel(model, device_ids=[0])

env_name = args.env_name

env = DecisionEnv(model, cfg)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = REINFORCE(args.hidden_size, len(env.actions))

dir = 'ckpt_' + env_name
if not os.path.exists(dir):
    os.mkdir(dir)

for i_episode in range(args.num_episodes):
    state = env.reset()
    # state = torch.Tensor([env.reset()])
    entropies = []
    log_probs = []
    rewards = []
    for t in range(args.num_steps):
        # agent.reset()
        if t == 0:
            action = 0
        else:
            action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()

        next_state, reward, done, _ = env.step(action.numpy()[0])

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state
        # state = torch.Tensor([next_state])

        # TODO 图片结束
        if done:
            break

    agent.update_parameters(rewards, log_probs, entropies, args.gamma)

    if i_episode % args.ckpt_freq == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'reinforce-' + str(i_episode) + '.pkl'))

    print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

# env.close()
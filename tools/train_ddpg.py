#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/20 下午3:14
# @Author  : FengDa
# @File    : train_ddpg.py
# @Software: PyCharm

# Training script for the DDPG
import random
import torch
# Add this line to get better performance
torch.backends.cudnn.benchmark=True
# from Utils import utils
import torch.optim as optim
from mmdet.models.decision_net.ddpg import DDPG
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
# from Training.trainer import Trainer
from mmdet.models.decision_net.ddpg_trainer import Trainer
import os

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

def parser_args():
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
    return args


if __name__ == '__main__':
    args = parser_args()

    # config
    cfg = mmcv.Config.fromfile(args.config)

    # build task model
    assert args.gpus == 1
    cfg.gpus = args.gpus

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint)
    device = torch.device("cuda")
    model.to(device)
    # model = MMDataParallel(model, device_ids=[0])

    env_name = args.env_name
    env = DecisionEnv(model, cfg)

    # Specify the environment name and create the appropriate environment
    seed = 41

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # env = utils.EnvGenerator(name='FetchReach-v1', goal_based=False, seed=seed)
    # eval_env = utils.EnvGenerator(name='FetchReach-v1', goal_based=False,seed=seed)
    # action_dim = env.get_action_dim()
    # observation_dim = env.get_observation_dim()
    # goal_dim =  env.get_goal_dim()
    # env= env.get_environment()
    # eval_env = eval_env.get_environment()

    # Training constants
    her_training=True
    # Future framnes to look at
    future= 0

    buffer_capacity = int(1e3)
    input_dim = 512
    action_dim = 1
    q_dim = 1
    batch_size = 128
    hidden_units = 256
    gamma = 0.98  # Discount Factor for future rewards
    num_epochs = 50
    learning_rate = 0.001
    critic_learning_rate = 0.001
    polyak_factor = 0.05
    # Huber loss to aid small gradients
    criterion = F.smooth_l1_loss
    # Adam Optimizer
    opt = optim.Adam

    # Output Folder
    output_folder = os.getcwd() + '/output_ddpg/'

    # Convert the observation and action dimension to int
    # print(observation_dim)
    # observation_dim = int(observation_dim)
    action_dim = int(action_dim)
    print(action_dim)
    # goal_dim= int(goal_dim)

    # Create the agent
    agent = DDPG(num_hidden_units=hidden_units, input_dim=input_dim,
                      num_actions=action_dim, num_q_val=q_dim, batch_size=batch_size, random_seed=seed,
                      use_cuda=use_cuda, gamma=gamma, actor_optimizer=opt, critic_optimizer=optim,
                      actor_learning_rate=learning_rate, critic_learning_rate=critic_learning_rate,
                      loss_function=criterion, polyak_constant=polyak_factor, buffer_capacity=buffer_capacity,
                 goal_dim=None, observation_dim=None)

    # Train the agent
    trainer = Trainer(agent=agent, num_epochs=50, num_rollouts=19*50, num_eval_rollouts=100,
                      max_episodes_per_epoch=50, env=env, eval_env=None,
                      nb_train_steps=19*50, multi_gpu_training=False, random_seed=seed, future=future)

    if her_training:
        trainer.her_training()
    else:
        trainer.train()
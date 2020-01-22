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
cfg.gpus = args.gpus

model = build_detector(
    cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
load_checkpoint(model, args.checkpoint)
device = torch.device("cuda")
model.to(device)
# model = MMDataParallel(model, device_ids=[0])

env_name = args.env_name

env = DecisionEnv(model, cfg)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

feat_channel = 512

import torch.optim as optim
from mmdet.models.decision_net.ddpg import DDPG
import torch.nn.functional as F
# Create the agent
# Training constants
her_training = True
# Future framnes to look at
future = 0

seed = 41
buffer_capacity = int(1e3)
input_dim = 512
action_dim = 1
q_dim = 1
batch_size = 8
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
use_cuda = torch.cuda.is_available()
# Output Folder
output_folder = os.getcwd() + '/output_ddpg/'
agent = DDPG(num_hidden_units=hidden_units, input_dim=input_dim,
             num_actions=action_dim, num_q_val=q_dim, batch_size=batch_size, random_seed=seed,
             use_cuda=use_cuda, gamma=gamma, actor_optimizer=opt, critic_optimizer=optim,
             actor_learning_rate=learning_rate, critic_learning_rate=critic_learning_rate,
             loss_function=criterion, polyak_constant=polyak_factor, buffer_capacity=buffer_capacity,
             goal_dim=None, observation_dim=None)
# agent = REINFORCE(feat_channel, 1)

dir = 'ckpt_' + env_name
if not os.path.exists(dir):
    os.mkdir(dir)


def her_training(env, ddpg):
    import time
    from collections import deque
    # Starting Time
    start_time = time.time()

    # Initialize the statistics dictionary
    # statistics = self.statistics

    episode_rewards_history = deque(maxlen=100)
    eval_episode_rewards_history = deque(maxlen=100)

    epoch_episode_rewards = []

    # Rewards and success for each epoch
    epoch_rewards = []

    # If eval, initialize the evaluation with an initial state
    # if eval_env is not None:
    #     eval_state = eval_env.reset()

    # Initialize the losses
    loss = 0
    episode_reward = 0
    episode_step = 0
    t = 0

    for epoch in range(10):
        epoch_actor_losses = []
        epoch_critic_losses = []

        for cycle in range(10):
            state = env.reset()

            # Rollout of trajectory to fill the replay buffer before the training
            for rollout in range(50):
                print('rollout:{}'.format(rollout))
                # Sample an action from behavioural policy pi
                if rollout == 0:
                    action = ddpg.get_action(state=state, noise=False)
                    action = np.clip(action, 0.75, 1)
                else:
                    action = ddpg.get_action(state=state, noise=True)

                # Execute the action and observe the new state
                new_state, reward, done = env.step(action)
                done_bool = done * 1
                # Store the transition in the experience replay
                ddpg.store_transition(
                        state=state, new_state=new_state, reward=reward, action=action, done=done_bool)

                t += 1
                episode_reward += reward
                episode_step += 1

                # Set the current state as the next state
                state = new_state

                # End of the episode
                if done:
                    episode_reward = 0
                    episode_step = 0
                    break
                    # Reset the agent
                    # self.ddpg.reset()
                    # Get a new initial state to start from
                    state = env.reset()

                # if rollout == 1:
                #     break
            # Train the network
            for train_steps in range(5):
                critic_loss, actor_loss = ddpg.fit_batch()
                if critic_loss is not None and actor_loss is not None:
                    epoch_critic_losses.append(critic_loss.cpu().detach().numpy())
                    epoch_actor_losses.append(actor_loss.cpu().detach().numpy())

                    print("epoch_critic_losses_mean:{}".format(np.mean(epoch_critic_losses)))
                    print("epoch_actor_losses_mean:{}".format(np.mean(epoch_actor_losses)))
                # Update the target networks using polyak averaging
                ddpg.update_target_networks()

            # TODO evaluation
            # eval_episode_rewards = []
            # eval_episode_successes = []
            # if self.eval_env is not None:
            #     eval_episode_reward = 0
            #     eval_episode_success = 0
            #     for t_rollout in range(self.num_eval_rollouts):
            #         if eval_state is not None:
            #             eval_action = self.ddpg.get_action(state=eval_state, noise=False)
            #         eval_new_state, eval_reward, eval_done, eval_success = self.eval_env.step(eval_action)
            #         eval_episode_reward += eval_reward
            #         eval_episode_success += eval_success['is_success']

            #         if eval_done:
            #             # Get the episode goal
            #             #eval_episode_goal = eval_new_state[:self.ddpg.obs_dim]
            #             #eval_episode_goals_history.append(eval_episode_goal)
            #             eval_state = self.eval_env.reset()
            #             eval_state = to_tensor(eval_state, use_cuda=self.cuda)
            #             eval_state = torch.unsqueeze(eval_state, dim=0)
            #             eval_episode_rewards.append(eval_episode_reward)
            #             eval_episode_rewards_history.append(eval_episode_reward)
            #             eval_episode_successes.append(eval_episode_success)
            #             eval_episode_success_history.append(eval_episode_success)
            #             eval_episode_reward = 0
            #             eval_episode_success = 0

            # Log stats
            duration = time.time() - start_time

        # Log the epoch rewards and successes
        epoch_rewards.append(np.mean(epoch_episode_rewards))
        torch.cuda.empty_cache()

    return


if __name__ == "__main__":
    her_training(env, agent)

    for i_episode in range(args.num_episodes):
        state = env.reset()
        # state = torch.Tensor([env.reset()])
        entropies = []
        log_probs = []
        rewards = []
        for t in range(args.num_steps):
            print(t)
            # agent.reset()
            if t == 0:
                action = 1
            else:
                action, action_prob = agent.select_action(state)
                # action = torch.tensor(1).cuda()
            # action = action.cpu()

            # next_state, reward, done, _ = env.step(action.numpy()[0])
            next_state, reward, done = env.step(action)

            # entropies.append(entropy)
            # log_probs.append(log_prob)
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
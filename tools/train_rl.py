#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/17 下午12:33
# @Author  : FengDa
# @File    : train_rl.py
# @Software: PyCharm

import argparse, math, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import random
import traceback
from collections import deque

import torch
import torch.optim as optim
from mmdet.models.decision_net.ddpg import DDPG
import torch.nn.functional as F

import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import scatter, collate, MMDataParallel
from mmdet.models import build_detector, detectors
# from normalized_actions import NormalizedActions
from mmdet.models.decision_net import DecisionEnv, DDPG, DQN
from mmdet.models.decision_net.policy_gradient import REINFORCE

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


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
    parser.add_argument('--checkpoint', default="../results/20200225-125037/epoch_11.pth", help='checkpoint file')
    parser.add_argument('--checkpointflow', default="../pretrained_models/flownetc_EPE1.766.tar",
                        help='checkpoint file')
    parser.add_argument(
        '--save_path', default="/home/ubuntu/datasets/YT-VIS/results/",
        type=str,
        help='path to save visual result')
    parser.add_argument('--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument('--proc_per_gpu', default=1, type=int, help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--load_result', default=False,
                        # action='store_true',
                        help='whether to load existing result')
    parser.add_argument(
        '--eval',
        default=['segm'],
        type=str,
        nargs='+',
        choices=['bbox', 'segm'],
        help='eval types')
    # parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show', default=True, help='show results')

    args = parser.parse_args()

    return args


def her_training(trn_env, val_env, agent, args):
    print("Algorithm:{}".format(agent.name()))
    # Starting Time
    start_time = time.time()

    # Initialize the statistics dictionary
    # statistics = self.statistics

    # episode_rewards_history = deque(maxlen=100)
    # eval_episode_rewards_history = deque(maxlen=100)
    #
    # epoch_episode_rewards = []

    # Rewards and success for each epoch
    # epoch_rewards = []

    # Initialize the losses
    loss = 0
    episode_reward = 0
    episode_step = 0
    t = 0

    for epoch in range(10):
        epoch_actor_losses = []
        epoch_critic_losses = []

        for cycle in tqdm(range(100)):
            try:
                state = trn_env.reset()
            except:
                print(traceback.print_exc())
                continue

            # Rollout of trajectory to fill the replay buffer before the training
            for rollout in range(100):
                try:
                    # print('rollout:{}'.format(rollout))
                    # Sample an action from behavioural policy pi
                    if rollout == 0:
                        # action = ddpg.get_action(state=state, noise=False)
                        # action = np.clip(action, 0.75, 1)
                        action = 0
                    else:
                        action = agent.select_action(state=state)
                        # action = np.argmax(action)

                    # Execute the action and observe the new state
                    new_state, reward, done = trn_env.step(action)
                    done_bool = done * 1
                    # Store the transition in the experience replay
                    agent.store_transition(state=state, new_state=new_state, reward=reward, action=action, done=done_bool)

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
                        # state = trn_env.reset()

                    # if rollout == 1:
                    #     break
                except:
                    print(traceback.print_exc())
                    continue
            # Train the network
            if agent.buffer.get_buffer_size() < agent.buffer.capacity:
                continue
            print("Updating parameters.")
            agent.learn()
            # for train_steps in range(10):
            #     critic_loss, actor_loss = agent.fit_batch()
            #     if critic_loss is not None and actor_loss is not None:
            #         epoch_critic_losses.append(critic_loss.cpu().detach().numpy())
            #         epoch_actor_losses.append(actor_loss.cpu().detach().numpy())
            #
            #         print("epoch_critic_losses_mean:{}".format(np.mean(epoch_critic_losses)))
            #         print("epoch_actor_losses_mean:{}".format(np.mean(epoch_actor_losses)))
            #     # Update the target networks using polyak averaging
            #     agent.update_target_networks()

        val_episode_reward = []
        # for idx in tqdm(range(len(val_videos))):
        print("Testing.")
        for idx in tqdm(range(len(val_videos))):
            try:
                state = val_env.reset(index=idx)
            except:
                print(traceback.print_exc())
                continue

            # Rollout of trajectory to fill the replay buffer before the training
            for rollout in range(200):
                # print('rollout:{}'.format(rollout))
                # Sample an action from behavioural policy pi
                if rollout == 0:
                    action = 0
                    # action = agent.get_action(state=state, noise=False)
                    # action = np.clip(action, 0.75, 1)
                else:
                    action = agent.select_action(state=state)
                    # action = agent.get_action(state=state, noise=False)

                # Execute the action and observe the new state
                new_state, reward, done = val_env.step(action)
                done_bool = done * 1
                # Store the transition in the experience replay
                # ddpg.store_transition(state=state, new_state=new_state, reward=reward, action=action, done=done_bool)

                val_episode_reward.append(reward)

                # Set the current state as the next state
                state = new_state

                # End of the episode
                if done:
                    break

            # Log stats
            duration = time.time() - start_time

        print("val_rewards_mean:{}".format(np.mean(val_episode_reward)))
        # Save model
        if args.save_model:
            # now = time.strftime("%Y-%m-%d %X",time.localtime())
            dir = os.path.join(args.output_folder, "epoch{}-{}".format(epoch, np.mean(val_episode_reward)))
            if not os.path.exists(dir):
                os.makedirs(dir)
            agent.save_model(dir)
        # Log the epoch rewards and successes
        # epoch_rewards.append(np.mean(val_episode_reward))
        torch.cuda.empty_cache()

    return


def testing(tst_env, tst_videos, ddpg, args):

    tst_episode_reward = []
    results = []

    # Starting Time
    start_time = time.time()

    for idx in tqdm(range(len(tst_videos))):
    # for idx in tqdm(range(200)):
        try:
            state = val_env.reset(index=idx)
        except:
            print(traceback.print_exc())
            continue

        # Rollout of trajectory to fill the replay buffer before the training
        for rollout in range(500):
            # print('rollout:{}'.format(rollout))
            # Sample an action from behavioural policy pi
            if rollout == 0:
                # action = ddpg.get_action(state=state, noise=False)
                # action = np.clip(action, 0.75, 1)
                action = 0.9
            else:
                action = ddpg.get_action(state=state, noise=False)

            # Execute the action and observe the new state
            new_state, reward, done = val_env.step(action)

            tst_episode_reward.append(reward)

            # Set the current state as the next state
            state = new_state

            # End of the episode
            if done:
                break

        # Log stats
        duration = time.time() - start_time

    print("val_rewards_mean:{}".format(np.mean(tst_episode_reward)))

    torch.cuda.empty_cache()

    return

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parser_args()
    # config
    cfg = mmcv.Config.fromfile(args.config)

    # build task model
    # assert args.gpus == 1
    cfg.gpus = args.gpus
    # Output Folder
    output_folder = os.getcwd() + '/output_ddpg/'
    now = time.strftime("%Y-%m-%d %X", time.localtime())
    cfg.output_folder = os.path.join(output_folder, now)
    cfg.save_model = True

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint)
    device = torch.device("cuda:0")
    model.to(device)
    # model = model.half()
    # model = MMDataParallel(model, device_ids=[0])
    # model = MMDataParallel(model)
    # put model on gpus
    # model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    # environments
    env_name = args.env_name

    trn_videos = list(pd.read_csv('train.csv').video_name)
    val_videos = list(pd.read_csv('val.csv').video_name)
    val_video = random.sample(val_videos, 50)
    # trn_videos = ['2c11fedca8']
    # val_videos = ['8ee2368f40']
    trn_env = DecisionEnv(model, cfg, trn_videos, is_train=True)
    val_env = DecisionEnv(model, cfg, val_video, is_train=False)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    feat_channel = 512

    # Create the agent
    # Training constants
    # her_training = True
    # Future framnes to look at
    future = 0

    seed = args.seed
    buffer_capacity = int(5000)
    input_dim = 512
    action_dim = 4
    q_dim = 1
    batch_size = 32
    hidden_units = 256
    gamma = 0.98  # Discount Factor for future rewards
    num_epochs = 50
    learning_rate = 0.00005
    critic_learning_rate = 0.00005
    polyak_factor = 0.05
    # Huber loss to aid small gradients
    criterion = F.smooth_l1_loss
    # Adam Optimizer
    opt = optim.Adam
    use_cuda = torch.cuda.is_available()

    # actor_pkl = "/home/ubuntu/code/fengda/MaskTrackRCNN/tools/output_ddpg" \
    #             "/2020-02-04 19:54:21/epoch0--0.09449123407716865/actor.pkl"
    # critic_pkl = "/home/ubuntu/code/fengda/MaskTrackRCNN/tools/output_ddpg" \
    #              "/2020-02-04 19:54:21/epoch0--0.09449123407716865/critic.pkl"
    actor_pkl = None
    critic_pkl = None


    # agent = DDPG(num_hidden_units=hidden_units, input_dim=input_dim,
    #              num_actions=action_dim, num_q_val=q_dim, batch_size=batch_size, random_seed=seed,
    #              use_cuda=use_cuda, gamma=gamma, actor_optimizer=opt, critic_optimizer=optim,
    #              actor_learning_rate=learning_rate, critic_learning_rate=critic_learning_rate,
    #              loss_function=criterion, polyak_constant=polyak_factor, buffer_capacity=buffer_capacity,
    #              goal_dim=None, observation_dim=None, a_check_point=actor_pkl, c_check_point=critic_pkl)

    agent = DQN(input_channel=input_dim, num_outputs=action_dim, batch_size=batch_size, random_seed=seed,
                  lr=learning_rate, gamma=gamma,  buffer_capacity=buffer_capacity)
    # agent = REINFORCE(feat_channel, 1)

    her_training(trn_env, val_env, agent, cfg)

    print('finish.')

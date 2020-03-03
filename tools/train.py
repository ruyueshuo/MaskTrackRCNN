from __future__ import division

import argparse
from mmcv import Config

from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default="../configs/masktrack_rcnn_r50_fpn_1x_flow_youtubevos.py",
                        help='train config file path')
    parser.add_argument('--work_dir', default="../results/",
                        help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from',
        # default=None,
        default = "/home/ubuntu/code/fengda/MaskTrackRCNN/results/20200225-125037/epoch_11.pth",
        # default="../results/20191213-113208/epoch_15.pth",
        # default="../pretrained_models/epoch_12.pth",
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        default=True,  # action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    import datetime
    now = datetime.datetime.now()
    now_str = datetime.datetime.strftime(now, "%Y%m%d-%H%M%S")
    args.work_dir = "../results/{}/".format(now_str)
    return args


def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus
    if cfg.checkpoint_config is not None:
        # save mmdet version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.text)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # get dataset
    from mmdet.models.decision_net.utils import modify_cfg
    video_name = '0043f083b5'
    cfg.data.train = modify_cfg(cfg, video_name)
    train_dataset = get_dataset(cfg.data.train)
    print("len of dataset: {}.".format(len(train_dataset)))

    # train
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    print(torch.__version__)
    main()

import argparse
import numpy as np

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json_videoseg, ytvos_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector, detectors
from mmdet.models.flow_heads.flownetC_head import FlowNetCHead
from mmdet.models.decision_net.utils import trans_action, change_img_meta, resize, modify_cfg, get_dataloader


def single_test(model, tst_videos, cfg, env=None, rl_model=None, show=False, save_path=''):
    model.eval()
    rl_model.eval()
    results = []

    prog_bar = mmcv.ProgressBar(len(tst_videos))

    scale_factors = [1, 1 / 2, 1 / 3, 1 / 4]

    for video_name in tst_videos:
        # Build dataloader
        cfg_val = modify_cfg(cfg, video_name, is_train=False)
        dataset = get_dataset(cfg_val)
        print('len of dataset ', len(dataset))
        data_loader = get_dataloader(dataset)

        for i, data in enumerate(data_loader):
            # Image Resolution is calculated by reinforcement learning algorithm.
            img_meta = data['img_meta'][0].data
            is_first = img_meta[0][0]['is_first']
            if is_first:
                key_frame = True
            else:
                key_frame = False
                state = None

                # Get Scale Factor
                action = rl_model.get_action(state)
                scale_factor = scale_factors[trans_action(action)]

                # Resize image according to scale factor. Impad to meet FPN requirement.
                data['img'][0] = resize(data['img'][0], scale_factor=scale_factor, size_divisor=32)

                # Change image meta info according to scale factor.
                data = change_img_meta(data, scale_factor=scale_factor)

            with torch.no_grad():
                result = model(return_loss=False, key_frame=key_frame, rescale=True, **data)
            results.append(result)

            show = False
            if show:
                model.module.show_result(data, result, dataset.img_norm_cfg,
                                         dataset=dataset.CLASSES,
                                         save_vis=True,
                                         save_path=save_path,
                                         is_video=True)

            batch_size = data['img'][0].size(0)
            for _ in range(batch_size):
                prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', default="../configs/masktrack_rcnn_r50_fpn_1x_flow_youtubevos.py",
                        help='test config file path')
    parser.add_argument('--checkpoint', default="../results/20191213-163326/epoch_16.pth", help='checkpoint file')
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

    import os
    args.save_path = os.path.dirname(args.checkpoint) + '/'

    return args


def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args()
    args.out = args.save_path + 'result_test.pkl'
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # get video list
    with open('valid.txt', "r") as f:  # 设置文件对象
        videos = f.read()  # 可以是随便对文件的操作
    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    # build model
    assert args.gpus == 1
    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint)
    model = MMDataParallel(model, device_ids=[0])

    # model_flow = FlowNetCHead()
    # load_checkpoint(model_flow, args.checkpointflow)

    if args.load_result:
        outputs = mmcv.load(args.out)

    else:
        # test
        outputs = single_test(model, videos, cfg, args.show, save_path=args.save_path)

    if args.out:
        if not args.load_result:
            print('writing results to {}'.format(args.out))

            mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if not isinstance(outputs[0], dict):
                result_file = args.out + '.json'
                results2json_videoseg(dataset, outputs, result_file)
                ytvos_eval(result_file, eval_types, dataset.ytvos)
            else:
                NotImplemented


if __name__ == '__main__':
    main()

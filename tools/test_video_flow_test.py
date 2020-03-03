#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/10 下午2:15
# @Author  : FengDa
# @File    : test_video_flow_test.py
# @Software: PyCharm
import argparse
import traceback
from tqdm import tqdm
import numpy as np

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json_videoseg, ytvos_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors

from mmdet.models.decision_net.utils import *

def single_test(model, data_loader, rl_model=None, show=False, save_path=''):
    model.eval()
    results = []
    scales = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    num = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    last_full = dict()

    for i, data in enumerate(data_loader):
        # if i == 0:
        #     img1 = data['img']
        #     continue
        # if i == 1:
        #     img2 = data['img']
        #
        # flow_network_data = torch.load("/home/ubuntu/code/fengda/MaskTrackRCNN/pretrained_models/flownetc_EPE1.766.tar")
        # print("=> using pre-trained model '{}'".format(flow_network_data['arch']))
        # from mmdet.models.flow_heads import FlowNetC, flownetc
        # flow_model = FlowNetC(batchNorm=False)
        # flow_model.load_state_dict(flow_network_data['state_dict'])
        # # flow_model = models.__dict__[flow_network_data['arch']](flow_network_data).cuda()
        # flow_model.cuda()
        # flow_model.eval()
        # flow_outputs2 = flow_model(img1[0].cuda(), img2[0].cuda())
        #
        # f_model = FlowNetC(checkpoint="/home/ubuntu/code/fengda/MaskTrackRCNN/pretrained_models/flownetc_EPE1.766.tar")
        # f_model.cuda()
        # flow_outputs3 = f_model(img1[0].cuda(), img2[0].cuda())
        # from mmdet.models.flow_heads.visualization import plot_flow
        # plot_flow(flow_outputs3[0], 'test.jpg')
        # data['img'] = data['img'][0]
        # data['img_meta'] = data['img_meta'][0].data[0]
        is_first = data['img_meta'][0].data[0][0]['is_first']

        if is_first:
            # Full resolution if it is the first frame of the video.
            scale_facotr = 1
            num = 1

            def state_reset():
                """Reset Features for states."""
                feat_self = get_self_feat(model, data['img'])
                feat_diff = torch.zeros_like(feat_self)
                feat_FAR = torch.tensor([0.]).cuda()
                feat_history = torch.zeros([10]).cuda()

                return [feat_diff, feat_self, feat_FAR, feat_history]

            state = state_reset()
        else:
            # If RL model is available, resolution is decided by rl model.
            if rl_model:
                def state_step(s, a):

                    return state
                state = state_step(state_prev, action_prev)
                scale_facotr = rl_model(state)
            # Or resolution is decided manually. e.g. One full resolution frame for every five frames.
            else:
                if num % 2 == 0:
                    scale_facotr = 1
                else:
                    scale_facotr = 0.5
                num += 1

        """
        # Test
        if is_first:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            last_data = data
            last_full['img'] = data['img'][0]
            last_full['feat_map_last'] = result[2]
        else:
            # scale_facotr = 0.5
            from mmdet.models.decision_net.utils import get_low_data
            from copy import deepcopy
            # with torch.no_grad():
            #     result_f = model(return_loss=False, rescale=True, **last_data)
            # last_full['img'] = last_data['img'][0]
            # last_full['feat_map_last'] = result_f[2]

            low_data = get_low_data(deepcopy(data), scale_facotr, size_divisor=32, is_train=False)
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, key_frame=last_full, **low_data)
            # last_data = data
        """

        if scale_facotr == 1:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            last_full['img'] = data['img'][0]
            last_full['feat_map_last'] = result[2]
        else:
            # resize data
            from mmdet.models.decision_net.utils import get_low_data
            from copy import deepcopy
            low_data = get_low_data(deepcopy(data), scale_facotr, size_divisor=32, is_train=False)
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, key_frame=last_full, **low_data)

            # # Check whether need to get higher resolution.
            # bboxes = result[0]
            # thr = 0.25
            # if len(bboxes) == 0:
            #     scale_facotr = 1
            # else:
            #     for key, val in bboxes.items():
            #         if min(val['bbox'][:-1]) < thr:
            #         # if min(bboxes[:-1]) < thr:
            #             scale_facotr = 1
            #         break
            # if scale_facotr == 1:
            #     with torch.no_grad():
            #         result = model(return_loss=False, rescale=True, **data)
            #     last_full['img_last_full'] = data['img']
            #     last_full['feat_last_full'] = result[2]

        print(scale_facotr)
        result = result[0:2]
        results.append(result)

        show = True
        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES,
                                     save_vis=True,
                                     save_path=save_path,
                                     rescale=True,
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
    parser.add_argument('--checkpoint', default="../results/20200225-125037/epoch_11.pth", help='checkpoint file')
    parser.add_argument('--checkpointflow', default="../pretrained_models/flownetc_EPE1.766.tar", help='checkpoint file')
    parser.add_argument(
        '--save_path', default="/home/ubuntu/datasets/YT-VIS/results/train-flow20-before-det/",
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
        default=['segm'],
        type=str,
        nargs='+',
        choices=['bbox', 'segm'],
        help='eval types')
    # parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show', default=True, help='show results')
    args = parser.parse_args()

    import os
    # args.save_path = os.path.dirname(args.checkpoint) + '/'
    # args.save_path = '/home/ubuntu/code/fengda/MaskTrackRCNN/results/test/flow/c71f30d7b6/full/'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args


def main():
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    # get dataset
    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    # build model
    assert args.gpus == 1
    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint)
    # device = torch.device("cuda")
    # model.to(device)
    model.load_flow()
    model = MMDataParallel(model, device_ids=[0])

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False)
    if args.load_result:
        outputs = mmcv.load(args.out)

    else:
        outputs = single_test(model, data_loader, show=args.show, save_path=args.save_path)

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
        # test
        Test = False
        if Test:
            import pandas as pd
            from mmdet.models.decision_net.utils import modify_cfg, get_dataloader
            from mmdet.datasets.utils import get_dataset
            # val_videos = list(pd.read_csv('train.csv').video_name)
            val_videos = ['2c11fedca8']
            from tqdm import tqdm
            for video_name in tqdm(val_videos):
                # get data loader of the selected video
                cfg_test = modify_cfg(cfg, video_name)
                ann_file = cfg_test.ann_file
                # self.dataset = obj_from_dict(self.cfg_test, datasets, dict(test_mode=True))
                try:
                    dataset = obj_from_dict(cfg_test, datasets, dict(test_mode=True))
                    print('video name: {}.\t len of dataset:{}.'.format(video_name, len(dataset)))
                    data_loader = build_dataloader(dataset, imgs_per_gpu=1, workers_per_gpu=0, num_gpus=1, dist=False, shuffle=False)

                    outputs = single_test(model, data_loader, cfg, args.show, save_path=os.path.join(args.save_path, video_name))
                except:
                    print(traceback.print_exc())
                    continue
                # break
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


def separate_annotations():
    """Separate train or validation annotations to single video annotation."""
    data_root = '/home/ubuntu/datasets/YT-VIS/'
    ann_file = data_root + 'annotations/instances_train_sub.json'
    import json
    with open(ann_file, 'r') as f:
        ann = json.load(f)
        # ann['videos'] = ann['videos'][15]
        # video_id = [0]
        from tqdm import tqdm
        for id in tqdm(range(len(ann['videos']))):
            videos = []
            anns = []
            video = ann['videos'][id]
            video['id'] = 1
            videos.append(video)

            i = 1
            for a in ann['annotations']:
                if a['video_id'] == id + 1:
                    anno = a
                    anno['id'] = i
                    anno['video_id'] = 1
                    anns.append(anno)
                    i += 1
            # anno = ann['annotations'][id]
            # anno['id'] = 1
            # anno['video_id'] = 1
            # anns.append(anno)

            file_name = videos[0]['file_names'][0].split('/')[0]

            ann_new = dict()
            ann_new['info'] = ann['info']
            ann_new['licenses'] = ann['licenses']
            ann_new['categories'] = ann['categories']
            ann_new['videos'] = videos
            ann_new['annotations'] = anns

            with open(data_root + 'train/Annotations/{}/{}_annotations.json'.format(file_name, file_name), 'w') as f:
                json.dump(ann_new, f, ensure_ascii=False)


if __name__ == '__main__':
    # videos_list = separate_annos(mode='train')
    # with open('valid.txt', 'w') as f:
    #     for v in videos_list:
    #         f.writelines(v)
    #         f.writelines("\n")

    main()

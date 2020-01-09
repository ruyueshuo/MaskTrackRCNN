import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json_videoseg, ytvos_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors
from mmdet.models.flow_heads.flownetC_head import FlowNetCHead


def single_test(model, data_loader, cfg, show=False, save_path=''):
    model.eval()
    results = []
    import numpy as np
    avg_time = np.zeros((1, 4))
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    num = 0
    for i, data in enumerate(data_loader):
        # The image is the key frame if it is the first frame of a video
        # or every 5 frame.
        img_meta = data['img_meta'][0].data
        is_first = img_meta[0][0]['is_first']
        if is_first or num % 5 == 0:
            key_frame = True
            num = 1
        else:
            key_frame = False

            # resize image according to scale factor.
            scale_factor = 0.5
            img = torch.nn.functional.interpolate(data['img'][0], scale_factor=scale_factor, mode='bilinear', align_corners=True)
            device = img.device
            img_np = np.transpose(img.squeeze().numpy(), (1, 2, 0))

            # The size of last feature map is 1/32(specially for resnet50) of the original image,
            # Impad to ensure that the original image size is multiple of 32.
            img_np = mmcv.impad_to_multiple(img_np, cfg.data.test.size_divisor)
            data['img'][0]= torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)

            # change image meta info.
            img_meta[0][0]['img_shape'] = (int(img_meta[0][0]['img_shape'][0] * scale_factor),
                                        int(img_meta[0][0]['img_shape'][1] * scale_factor), 3)
            img_meta[0][0]['pad_shape'] = img_np.shape
            img_meta[0][0]['scale_factor'] = img_meta[0][0]['scale_factor'] * scale_factor

            num += 1

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

    # get dataset
    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    # build model
    assert args.gpus == 1
    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint)
    model = MMDataParallel(model, device_ids=[0])

    # model_flow = FlowNetCHead()
    # load_checkpoint(model_flow, args.checkpointflow)

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False)
    if args.load_result:
        outputs = mmcv.load(args.out)
        # import json
        # with open(args.out+'.json', "w") as f:
        #     json.dump(outputs, f)
    else:
        # test
        outputs = single_test(model, data_loader, cfg, args.show, save_path=args.save_path)

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
    # data_root = '/home/ubuntu/datasets/YT-VIS/'
    # ann_file = data_root + 'annotations/instances_train_sub.json'
    # import json
    # with open(ann_file, 'r') as f:
    #     ann = json.load(f)
    #     # ann['videos'] = ann['videos'][15]
    #     video_id = [0, 1, 2, 3, 4]
    #     videos = []
    #     anns =[]
    #     for id in video_id:
    #         videos.append(ann['videos'][id])
    #         anns.append(ann['annotations'][id])
    #     ann['videos'] = videos
    #     ann['annotations'] = anns
    #     # videos = ann['videos'][video_id]
    #     # ann['videos'] = []
    #     # ann['videos'].append(videos)
    #     # anns = ann['annotations'][video_id]
    #     # ann['annotations'] = []
    #     # ann['annotations'].append(anns)
    #
    # with open(data_root + 'annotations/instances_test_sub.json', 'w') as f:
    #     json.dump(ann, f, ensure_ascii=False)

    # from mmdet.models.mask_heads.res5_mask_head import ResMaskHead
    # model = ResMaskHead()
    # print(model)
    main()

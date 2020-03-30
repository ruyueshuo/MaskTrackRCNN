#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/11 15:48
# @Author  : FengDa
# @File    : visualization.py
# @Software: PyCharm

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 下午2:03
# @Author  : FengDa
# @File    : flow_utils.py
# @Software: PyCharm
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from mmdet.models.flow_heads.flownetC import FlowNetC
from imageio import imread, imwrite


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()


def flow2rgb(flow_map, max_value):
    """flow to rgb images files"""
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)


def cal_flow(img1_file, img2_file, flow_model, save_path,
             div_flow=20, output_value='vis', max_flow=20, plot=True):
    # Data loading code
    input_transform = transforms.Compose([
        ArrayToTensor(),
        # transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        # transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
        transforms.Normalize(mean=[104.805, 110.16, 114.75], std=[255, 255, 255])
        # transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    ])

    img1 = input_transform(imread(img1_file))
    img2 = input_transform(imread(img2_file))
    input_var = torch.cat([img1, img2]).unsqueeze(0)
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)


    size = None
    # size = (90, 160)
    if size is not None:
        img1 = F.interpolate(img1, size=size, mode='bilinear', align_corners=True)
        img2 = F.interpolate(img2, size=size, mode='bilinear', align_corners=True)

    # compute output
    output = flow_model(img1, img2)
    # output = model(img1, img2)[0]
    """
    # Data loading code
    input_transform1 = transforms.Compose([
        ArrayToTensor(),
        transforms.Normalize(mean=[104.805, 110.16, 114.75], std=[255, 255, 255])
        # transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    ])
    img11 = input_transform1(imread(img1_file))
    img21 = input_transform1(imread(img2_file))
    input_var = torch.cat([img11, img21]).unsqueeze(0)
    img11 = img11.unsqueeze(0).to(device)
    img21 = img21.unsqueeze(0).to(device)
    # compute output
    output1 = model(img11, img21)[0]

    diff = (output - output1).cpu().detach().numpy()
    output_np = output.cpu().detach().numpy()
    output1_np = output1.cpu().detach().numpy()
    # return
    """
    upsampling = None
    if upsampling is not None:
        output = F.interpolate(output, size=img1.size()[-2:], mode=upsampling, align_corners=False)

    if plot:
        filename = os.path.join(save_path, '{}-{}'.format(os.path.basename(img1_file)[:-4], 'flow'))
        plot_flow(output, filename, output_value='both', div_flow=20, max_flow=20)
    return True


def plot_flow(output, filename, output_value='vis', div_flow=20, max_flow=None):
    for suffix, flow_output in zip(['flow', 'inv_flow'], output):
        # tmp = os.path.basename(img1_file)
        # filename = os.path.join(save_path, '{}-{}'.format(os.path.basename(img1_file)[:-4], suffix))
        if output_value in ['vis', 'both']:
            rgb_flow = flow2rgb(div_flow * flow_output, max_value=max_flow)
            to_save = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
            imwrite(filename + '.png', to_save)
        if output_value in ['raw', 'both']:
            # Make the flow map a HxWx2 array as in .flo files
            to_save = (div_flow * flow_output.detach()).cpu().numpy().transpose(1, 2, 0)
            to_comp = np.load("/home/ubuntu/datasets/YT-VIS/flow/003234408d/00005-flow.npy")
            diff = to_save - to_comp
            np.save(filename + '.npy', to_save)


def folder_flow(img_folder, save_folder, flow_model):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    imgs = os.listdir(img_folder)
    imgs.sort()
    img1_file = os.path.join(img_folder, imgs[0])
    for idx, img in enumerate(imgs):
        if idx%5 == 0:
            img1_file = os.path.join(img_folder, imgs[idx - 5])
            img2_file = os.path.join(img_folder, img)
            # print(img1_file, img2_file)
            cal_flow(img1_file, img2_file, flow_model, save_folder)


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = "/home/ubuntu/code/fengda/MaskTrackRCNN/pretrained_models/flownetc_EPE1.766.tar"
    flow_model = FlowNetC(batchNorm=False, checkpoint=checkpoint)
    flow_model.to(device)
    flow_model.eval()
    path = "/home/ubuntu/datasets/YT-VIS/results/train"
    # file_names = os.listdir(path)
    # file_names.sort()
    # from tqdm import tqdm
    # for file in tqdm(file_names):
    #     # print(file)
    #     save_folder = os.path.join("/home/ubuntu/datasets/YT-VIS/results/flow-max/", file)
    #     img_folder = os.path.join("/home/ubuntu/datasets/YT-VIS/train/JPEGImages/", file)
    #     folder_flow(img_folder, save_folder, flow_model)

    save_path = "/home/ubuntu/datasets/YT-VIS/flow/003234408d/"
    # # img_folder = "/home/ubuntu/datasets/YT-VIS/train/JPEGImages/05a0a513df"
    #
    img1_file = "/home/ubuntu/datasets/YT-VIS/train/JPEGImages/003234408d/00000.jpg"
    img2_file = "/home/ubuntu/datasets/YT-VIS/train/JPEGImages/003234408d/00005.jpg"
    cal_flow(img2_file, img1_file, flow_model, save_path)

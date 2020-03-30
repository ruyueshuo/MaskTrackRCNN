#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-01-02 14:08
# @Author  : Zhaoyy
# @File    : propagation_utils.py
# @Software: PyCharm

'''
modified by zhaoyy on 2020-03-10
1. improve efficiency of warp
2. fixed a bug
'''
import torch
import numpy as np
import torch.nn.functional as F
from numpy import *
CUDA_LAUNCH_BLOCKING = 1
from copy import deepcopy


def get_batchindex(b, h, w):
    index_list = []
    for k in range(b):
        for i in range(h):
            for j in range(w):
                index_list.append([k])
    return np.array(index_list)

def get_batchindex_improved(b, h, w):
    index_list = []
    for k in range(b):
        index_list.extend([[k]] * h*w)
    return np.array(index_list)

def get_batchindex_improved2(b, h, w):

    index_list = [ [k]*h*w for k in range(b)]
    return np.array(index_list).reshape(b*h*w, -1)

def torch_gather_nd(images, coords):
    '''
    reimplement gather_nd in tensorflow using pytorch
    :param x:
    :param coords:
    :return:
    '''
    idx1, idx2, idx3 = coords.chunk(3, dim=3)
    x_gather = images[idx1, idx2, idx3]
    x_gather = x_gather.squeeze(3)
    return x_gather

def warp(key_feature, flow):
    '''
    feature interpolation from key feature to non-key feature by using feature flow method
    :param key_feature: the key frame's features, with size of batchsize * channels * h * w
    :param flow: the motion from key frame to non-key frame, with size of batchsize * 2 * h * w
    :return:  non-key feature, tesnsor with size of batchsize * channels * h * w
    '''
    key_feature.cuda()
    flow.cuda()

    key_feature_height, key_feature_width = key_feature.shape[-2:]

    flow = flow.permute([0, 2, 3, 1]).contiguous().cuda()  # 便于gather   [1, 180, 320, 2]

    # makes the key feature's size is equal to the flow's size
    batch_size, flow_height, flow_width, _ = flow.size()

    # resize the features's size as the same to the flow
    key_feature = F.interpolate(key_feature, (flow_height, flow_width), mode='bilinear', align_corners=True)

    '''
    可视化key feature resize 之后的效果
    
    print(key_feature.size(), flow.size())
    sampled_image = key_feature.squeeze().cpu()#[221, 206 , 190]
    sampled_image = np.transpose(sampled_image.data.numpy(), (1,2,0))
    sampled_image = sampled_image * 255
    cv2.imwrite('before_warped.jpg', sampled_image)
    '''

    # 生成raw location
    base_index_x, base_index_y = torch.meshgrid([torch.arange(flow_height), torch.arange(flow_width)])
    raw_location = torch.stack([base_index_y,base_index_x], -1).view(flow_height, flow_width, 2).cuda().float()
    raw_location.requires_grad = True

    key_feature = key_feature.permute([0, 2, 3, 1]).contiguous().cuda()  # 便于检索 [1, 180, 320, 3]

    # 移动之后的location
    flow_index = raw_location - flow
    # flow_index = deepcopy(_flow_index)
    # motion的位置强制不超过image的边界
    flow_index[:, :, :, 0] = flow_index[:, :, :, 0].clamp(0., flow_width - 1)
    flow_index[:, :, :, 1] = flow_index[:, :, :, 1].clamp(0., flow_height - 1)
    # a = torch.clamp(flow_index[:, :, :, 0], 0., flow_width - 1)
    # b = torch.clamp(flow_index[:, :, :, 1], 0., flow_height - 1)
    # flow_index[:, :, :, 0] = a
    # flow_index[:, :, :, 1] = b

    x_index = flow_index[:, :, :, 0].reshape(batch_size, flow_height, flow_width, 1)
    y_index = flow_index[:, :, :, 1].reshape(batch_size, flow_height, flow_width, 1)


    x_floor = torch.floor(x_index)
    x_ceil = torch.ceil(x_index)
    y_floor = torch.floor(y_index)
    y_ceil = torch.ceil(y_index)

    batch_index = get_batchindex_improved2(batch_size, flow_height, flow_width)
    batch_index = batch_index.reshape(batch_size, flow_height, flow_width, 1)
    batch_index = torch.from_numpy(batch_index).cuda().float()

    raw_location = torch.stack([raw_location for _ in range(batch_size)]).float().cuda()

    flow_index_ff = torch.cat((batch_index, y_floor, x_floor), 3).long()
    flow_index_cf = torch.cat((batch_index, y_ceil, x_floor), 3).long()
    flow_index_fc = torch.cat((batch_index, y_floor, x_ceil), 3).long()
    flow_index_cc = torch.cat((batch_index, y_ceil, x_ceil), 3).long()

    # get weight
    thetax = x_index - x_floor
    _thetax = torch.tensor([1.]).cuda() - thetax
    thetay = y_index - y_floor
    _thetay = torch.tensor([1.]).cuda() - thetay

    coeff_ff = _thetax * _thetay
    coeff_cf = _thetax * thetay
    coeff_fc = thetax * _thetay
    coeff_cc = thetax * thetay

    ff = torch_gather_nd(key_feature, flow_index_ff) * coeff_ff
    cf = torch_gather_nd(key_feature, flow_index_cf) * coeff_cf
    fc = torch_gather_nd(key_feature, flow_index_fc) * coeff_fc
    cc = torch_gather_nd(key_feature, flow_index_cc) * coeff_cc

    warp_image = ff + cf + fc + cc


    # 格式转换
    warp_image = warp_image.permute([0, 3, 1, 2]).contiguous()  # 便于检索
    '''
    可视化warp feature 采样之后的效果
    sampled_image = warp_image.squeeze().cpu()#[221, 206 , 190]
    sampled_image = np.transpose(sampled_image.data.numpy(), (1,2,0))
    sampled_image = sampled_image * 255
    cv2.imwrite('ours_warped.jpg', sampled_image)
    '''

    warp_image = F.interpolate(warp_image, (key_feature_height, key_feature_width), mode='bilinear', align_corners=True)

    del key_feature, raw_location, batch_index, flow_index, flow_index_cc, flow_index_cf,\
        flow_index_fc, flow_index_ff, ff, cf, fc, cc

    return warp_image

if __name__ == '__main__':

    # test scucessive warp

    import cv2
    import torchvision.transforms as trainsforms
    import os

    print('start...')
    transform = trainsforms.ToTensor()

    # 测试4083cfbe15 滑板车出现及移动---> 滑板车相邻5个frames移动比较大，但是相邻2frames可以看出移动
    video_name = '4083cfbe15'
    image_name_list = ['00045','00046','00047','00048','00049','00050','00051','00052',
                       '00052','00054','00055']
    image_name_list = ['00050','00051','00052',
                       '00052','00054','00055']

    # 测试0065b171f9--> 大熊猫吃竹子，6-10 frames移动较慢，10-17，竹子向下段了
    # video_name = '0065b171f9'
    # image_name_list = ['00006','00007','00008','00009','00010','00011','00012','00013',
    #                    '00014','00015','00016','00017']

    # 测试01c76f0a82--> 汽车拐弯较慢
    # video_name = '01c76f0a82'
    # image_name_list = ['00040','00041','00042','00043','00044','00045','00046','00047',
    #                    '00048','00049','00050']


    for img_name in image_name_list:


        # 读取原始image
        image = cv2.imread('/home/ubuntu/datasets/YT-VIS/train/JPEGImages/'+video_name+'/'+img_name+'.jpg')
        print('/home/ubuntu/datasets/YT-VIS/train/JPEGImages/'+video_name+'/'+img_name+'.jpg')
        image_tensor = transform(image)
        key_feature = torch.unsqueeze(image_tensor, 0)  # adding batch dimention [1, 3, 720, 1280 ]

        # 读取真实的flow的值
        print('/home/ubuntu/datasets/YT-VIS/flow/'+video_name+'/'+ img_name+'flow.npy')
        flow = np.load('/home/ubuntu/datasets/YT-VIS/flow/'+video_name+'/'+ img_name+'flow.npy')
        flow = np.array(flow)
        flow = transform(flow)
        flow = torch.unsqueeze(flow, 0)  # adding batch dimention  #[1, 2, 180, 320]

        # save_path
        save_path = '/home/ubuntu/datasets/YT-VIS/flow/warp/'+video_name+'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
         # 测试warp函数
        next_image_name = "%05d"%(int(img_name)+1)
        non_key_feature = warp(key_feature, flow)

        # test gather_nd
        # key_feature = F.interpolate(key_feature, (flow.size()[2], flow.size()[3]), mode='bilinear', align_corners=True)
        # non_key_feature = gather_nd(key_feature, flow)
        break

    # '''
    # test single warp
    # '''
    # import cv2
    # import torchvision.transforms as trainsforms
    # transform = trainsforms.ToTensor()
    #
    # # 读取一个image， 假设它就是key feature
    # image = cv2.imread('/home/ubuntu/datasets/YT-VIS/train/JPEGImages/4083cfbe15/00001.jpg')
    # cv2.imwrite('00005.jpg', image)
    #
    # image = cv2.imread('/home/ubuntu/datasets/YT-VIS/train/JPEGImages/4083cfbe15/00000.jpg')
    # cv2.imwrite('00000.jpg', image)
    #
    # image_tensor = transform(image)
    # key_feature = torch.unsqueeze(image_tensor, 0) # adding batch dimention [1, 3, 720, 1280 ]
    #
    # # 读取真实的flow的值
    # flow = np.load('/home/ubuntu/datasets/YT-VIS/flow/00000flow.npy')
    # flow = np.array(flow)
    # flow = transform(flow)
    # flow = torch.unsqueeze(flow, 0) # adding batch dimention  #[1, 2, 180, 320]
    #
    # # 测试warp函数
    # non_key_feature = warp(key_feature, flow)
    #
    # print('the non-key feature shape is {}'.format(non_key_feature.shape))
    # '''
    # test function get_xyindex_improved() and get_batchindex_improved(b, h, w)
    # '''
    #
    #
    # b = 2
    # h = 3
    # w = 7
    # test1 = get_batchindex(b, h, w)
    # test2 = get_batchindex_improved(b, h, w)
    # print(test2.shape)
    # test3 = get_batchindex_improved2(b, h, w)
    # print('..',test3, test3.shape)
    #
    # test1 = get_xyindex(h, w)
    # test2 = get_xyindex_improved(h, w)
    # print('test2',test2,test2.shape)
    # test3 = get_xyindex_improved2(h, w)
    # print('test3', test3.shape,test3)
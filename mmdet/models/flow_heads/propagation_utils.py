#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-01-02 14:08
# @Author  : Zhaoyy
# @File    : propagation_utils.py
# @Software: PyCharm
import torch
import numpy as np
import torch.nn.functional as F


def get_xyindex(h, w):
    index_list = []
    for i in range(h):
        for j in range(w):
            index_list.append([j, i])
    return np.array(index_list)


def get_batchindex(b, h, w):
    index_list = []
    for k in range(b):
        for i in range(h):
            for j in range(w):
                index_list.append([k])
    return np.array(index_list)


def torch_gather_nd(images, coords):
    '''
    reimplement gather_nd in tensorflow using pytorch
    :param x:
    :param coords:
    :return:
    '''

    idx1, idx2, idx3 = coords.chunk(3, dim=3)
    x_gather = images[idx1, idx2, idx3].squeeze(3)

    return x_gather


def warp(key_feature, flow):
    '''
    feature interpolation from key feature to non-key feature by using feature flow method
    :param key_feature: the key frame's features, with size of batchsize * channels * h * w
    :param flow: the motion from key frame to non-key frame, with size of batchsize * 2 * h * w
    :return:  non-key feature, tesnsor with size of batchsize * channels * h * w
    '''
    flow = flow.permute([0, 2, 3, 1]).contiguous()  # 便于检索

    # makes the key feature's size is equal to the flow's size
    batch_size, height, weight, _ = flow.size()
    key_feature = F.interpolate(key_feature, (height, weight), mode='bilinear', align_corners=True)

    key_feature = key_feature.permute([0, 2, 3, 1]).contiguous()  # 便于检索

    # 生成raw location
    raw_location = get_xyindex(height, weight)
    raw_location = np.array(raw_location).reshape((height, weight, 2))
    raw_location = torch.from_numpy(raw_location)

    device = flow.device
    raw_location = raw_location.to(device).float()
    # 移动之后的location
    flow_index = flow + raw_location

    # motion的位置强制不超过image的边界
    max_location = torch.Tensor([weight - 1, height - 1]).to(device)
    flow_index = torch.min(flow_index, max_location)

    min_location = torch.Tensor([0, 0]).to(device)
    flow_index = torch.max(flow_index, min_location)

    x_index = flow_index[:, :, :, 0].reshape(batch_size, height, weight, 1)
    y_index = flow_index[:, :, :, 1].reshape(batch_size, height, weight, 1)

    #
    x_floor = torch.floor(x_index)
    x_ceil = torch.ceil(x_index)
    y_floor = torch.floor(y_index)
    y_ceil = torch.ceil(y_index)

    #
    batch_index = get_batchindex(batch_size, height, weight)
    batch_index = np.array(batch_index).reshape(batch_size, height, weight, 1)
    batch_index = torch.from_numpy(batch_index).to(device).float()

    flow_index_ff = torch.cat((batch_index, y_floor, x_floor), 3).long()
    flow_index_cf = torch.cat((batch_index, y_ceil, x_floor), 3).long()
    flow_index_fc = torch.cat((batch_index, y_floor, x_ceil), 3).long()
    flow_index_cc = torch.cat((batch_index, y_ceil, x_ceil), 3).long()

    # get weight
    thetax = x_index - x_floor
    _thetax = 1.0 - thetax
    thetay = y_index - y_floor
    _thetay = 1.0 - thetay

    coeff_ff = _thetax * _thetay
    coeff_cf = _thetax * thetay
    coeff_fc = thetax * _thetay
    coeff_cc = thetax * thetay

    # import torchsample

    ff = torch_gather_nd(key_feature, flow_index_ff) * coeff_ff
    cf = torch_gather_nd(key_feature, flow_index_cf) * coeff_cf
    fc = torch_gather_nd(key_feature, flow_index_fc) * coeff_fc
    cc = torch_gather_nd(key_feature, flow_index_cc) * coeff_cc

    warp_image = ff + cf + fc + cc

    # 格式转换
    warp_image = warp_image.permute([0, 3, 1, 2]).contiguous()  # 便于检索

    return warp_image


if __name__ == '__main__':
    '''
    test warp  
    '''
    # import cv2
    # from utils import image_convert
    #
    # # 测试，假设生成一个key frame的feature map
    # image = cv2.imread('test_image.jpg')
    # image_tensor = image_convert.convert_image_to_tensor(image)
    # key_feature = torch.unsqueeze(image_tensor, 0)
    #
    # # 测试， 假设生成一个flow
    # # flow = key_feature.permute([0,2,3,1]).contiguous()
    # flow = key_feature[:, :2, :, :]
    # print('the flow shape is :', flow.shape)
    #
    # # 测试warp函数
    # non_key_feature = warp(key_feature, flow)
    #
    # print('the non-key feature shape is {}'.format(non_key_feature.shape))

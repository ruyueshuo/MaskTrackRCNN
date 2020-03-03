#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-01-02 14:08
# @Author  : Zhaoyy
# @File    : propagation_utils.py
# @Software: PyCharm
import torch
import numpy as np
import torch.nn.functional as F

CUDA_LAUNCH_BLOCKING = 1
def get_xyindex(h, w):
    index_list = []
    for i in range(h):
        for j in range(w):
            index_list.append([j, i])

    return np.array(index_list)

def get_xyindex_improved(h,w):
    '''
    reimplement get_xyindex to make it more efficient
    :param h:
    :param w:
    :return:
    '''

    index_list = []
    list_a = list(range(w))
    for i in range(h):
        tmp_list = [i] * w
        zipped = zip(list_a, tmp_list)
        index_list.extend(zipped)

    return np.array(index_list)

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
    if len(flow.shape) != 4:
        flow = flow.unsqueeze(0)
    device = flow.device
    dtype = flow.dtype

    # flow = F.interpolate(flow, key_feature.shape[-2:], mode='bilinear', align_corners=True)

    flow = flow.permute([0, 2, 3, 1]).contiguous()  # 便于检索
    orig_size = key_feature.shape[-2:]
    # makes the key feature's size is equal to the flow's size
    batch_size, height, weight, _ = flow.size()
    key_feature = F.interpolate(key_feature, (height, weight), mode='bilinear', align_corners=True)

    key_feature = key_feature.permute([0, 2, 3, 1]).contiguous()  # 便于检索

    # 生成raw location
    # raw_location = get_xyindex(height, weight)
    raw_location = get_xyindex_improved(height, weight)
    raw_location = np.array(raw_location).reshape((height, weight, 2))
    raw_location = torch.from_numpy(raw_location)
    raw_location = raw_location.to(device)
    raw_location = raw_location.float()
    # base_index_x, base_index_y = torch.meshgrid([torch.arange(flow.size()[1]), torch.arange(flow.size()[2])])
    # base_index = torch.stack([base_index_x, base_index_y], -1).view(flow.size()[1], flow.size()[2], 2).to(device)
    # base_index = torch.stack([base_index for _ in range(flow.size()[0])])
    # # raw_location = base_index.to(device)
    # raw_location = base_index.float()
    # 移动之后的location
    flow_index = flow + raw_location

    # motion的位置强制不超过image的边界
    max_location = torch.Tensor([weight - 1, height - 1]).to(device)
    flow_index = torch.min(flow_index, max_location)

    min_location = torch.Tensor([0, 0]).to(device)
    flow_index = torch.max(flow_index, min_location)

    x_index = flow_index[:, :, :, 0].reshape(batch_size, height, weight, 1)
    y_index = flow_index[:, :, :, 1].reshape(batch_size, height, weight, 1)

    # y_index = flow_index[:, :, :, 0].reshape(batch_size, height, weight, 1)
    # x_index = flow_index[:, :, :, 1].reshape(batch_size, height, weight, 1)
    #
    x_floor = torch.floor(x_index)
    x_ceil = torch.ceil(x_index)
    y_floor = torch.floor(y_index)
    y_ceil = torch.ceil(y_index)

    #
    # batch_index = get_batchindex(batch_size, height, weight)
    batch_index = get_batchindex_improved(batch_size, height, weight)
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

    warp_image = F.interpolate(warp_image, orig_size, mode='bilinear', align_corners=True)

    del key_feature, raw_location, max_location, min_location, batch_index, flow_index, flow_index_cc, flow_index_cf,\
        flow_index_fc, flow_index_ff, ff, cf, fc, cc
    # torch.cuda.empty_cache()
    return warp_image


def warp1(key_feature, flow):
    '''
    feature interpolation from key feature to non-key feature by using feature flow method
    :param key_feature: the key frame's features, with size of batchsize * channels * h * w
    :param flow: the motion from key frame to non-key frame, with size of batchsize * 2 * h * w
    :return:  non-key feature, tesnsor with size of batchsize * channels * h * w
    '''
    flow = flow.permute([0, 2, 3, 1]).contiguous()  # 便于检索
    orig_size = key_feature.shape[-2:]
    # makes the key feature's size is equal to the flow's size
    batch_size, height, weight, _ = flow.size()
    key_feature = F.interpolate(key_feature, (height, weight), mode='bilinear', align_corners=True)

    key_feature = key_feature.permute([0, 2, 3, 1]).contiguous()  # 便于检索

    # 生成raw location
    # raw_location = get_xyindex(height, weight)
    raw_location = get_xyindex_improved(height, weight)
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
    # batch_index = get_batchindex(batch_size, height, weight)
    batch_index = get_batchindex_improved(batch_size, height, weight)
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

    warp_image = F.interpolate(warp_image, orig_size, mode='bilinear', align_corners=True)

    return warp_image


def gather_nd(input, gather_index):
    # input: [batch_size, channels, height, width], gather_index: [batch_index, 2, height, width]
    input.cuda()
    gather_index.cuda()
    base_index_x, base_index_y = torch.meshgrid([torch.arange(input.size()[2]), torch.arange(input.size()[3])])
    base_index = torch.stack([base_index_x, base_index_y], -1).view(input.size()[2], input.size()[3], 2).cuda()
    base_index = torch.stack([base_index for _ in range(input.size()[0])]).double().cuda()

    input = input.permute(0, 2, 3, 1).contiguous().double()
    gather_index = gather_index.permute(0, 2, 3, 1).contiguous().double()
    gather_index = base_index + gather_index
    gather_index = gather_index.view(-1, 2).double()
    clamp_gather_index = torch.DoubleTensor(gather_index.size()).cuda()
    clamp_gather_index[:, 0] = torch.clamp(gather_index[:, 0], 0., float(input.size()[1] - 1)).double()
    clamp_gather_index[:, 1] = torch.clamp(gather_index[:, 1], 0., float(input.size()[2] - 1)).double()
    gather_index_ceil = torch.ceil(clamp_gather_index).double()
    gather_index_floor = torch.floor(clamp_gather_index).double()

    output = []
    for i in range(gather_index.size()[0]):
        batch_index = i // (input.size()[1] * input.size()[2])

        cor_x, cor_y = clamp_gather_index[i][0], clamp_gather_index[i][1]
        cor_x_ceil, cor_y_ceil = gather_index_ceil[i][0], gather_index_ceil[i][1]
        cor_x_floor, cor_y_floor = gather_index_floor[i][0], gather_index_floor[i][1]
        weight_ceil_x, weight_ceil_y = cor_x - cor_x_floor, cor_y - cor_y_floor
        weight_floor_x, weight_floor_y = cor_x_ceil - cor_x, cor_y_ceil - cor_y

        cor_x_ceil = cor_x_ceil.int()
        cor_y_ceil = cor_y_ceil.int()
        cor_x_floor = cor_x_floor.int()
        cor_y_floor = cor_y_floor.int()
        output_ceil = input[batch_index, cor_x_ceil, cor_y_ceil]
        output_floor = input[batch_index, cor_x_floor.int(), cor_y_floor.int()]
        output_y_ceil = weight_ceil_x * input[batch_index, cor_x_ceil.int(), cor_y_ceil.int()] + weight_floor_x * input[batch_index, cor_x_floor.int(), cor_y_ceil.int()]
        output_y_floor = weight_ceil_x * input[batch_index, cor_x_ceil.int(), cor_y_floor.int()] + weight_floor_x * input[batch_index, cor_x_floor.int(), cor_y_floor.int()]
        output.append(weight_ceil_y * output_y_ceil + weight_floor_y * output_y_floor)

    result = torch.stack(output, 0).view(tuple(input.size())).permute(0, 3, 1, 2).contiguous().float()

    return result


if __name__ == '__main__':
    '''
    test warp  
    '''
    import cv2
    from utils import image_convert

    # 测试，假设生成一个key frame的feature map
    image = cv2.imread('test_image.jpg')
    image_tensor = image_convert.convert_image_to_tensor(image)
    key_feature = torch.unsqueeze(image_tensor, 0)

    # 测试， 假设生成一个flow
    # flow = key_feature.permute([0,2,3,1]).contiguous()
    flow = key_feature[:, :2, :, :]
    print('the flow shape is :', flow.shape)

    # 测试warp函数
    non_key_feature = warp(key_feature, flow)

    print('the non-key feature shape is {}'.format(non_key_feature.shape))

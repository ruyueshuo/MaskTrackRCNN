#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/7 10:53
# @Author  : FengDa
# @File    : utils.py
# @Software: PyCharm
import PIL.Image as Image
import os


# 定义图像拼接函数
def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图


# 定义图像拼接函数
def img_compose(img1, img2, img3, img4):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    # for y in range(1, IMAGE_ROW + 1):
    #     for x in range(1, IMAGE_COLUMN + 1):
    # from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
    #             (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    # imgs = [img1, img2, img3, img4]
    # for img
    to_image.paste(img1, (0, 0))
    to_image.paste(img2, (IMAGE_SIZE, 0))
    to_image.paste(img3, (0, IMAGE_SIZE))
    to_image.paste(img4, (IMAGE_SIZE, IMAGE_SIZE))

    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图


def img_compose(orig_img, mask_img, save_path):
    import cv2
    import numpy as np
    img = cv2.imread(orig_img, 1)
    mask = cv2.imread(mask_img, 0)
    img = cv2.resize(img, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

    img = img[:, :, ::-1]
    img[..., 2] = np.where(mask == 1, 255, img[..., 2])

    cv2.imwrite(save_path, img)
    return True


def folder_compose(mask_folder, orig_folder, save_folder, anno=False):
    """将两个文件夹中的原图和对应的分割结果画出来。"""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    masks = os.listdir(mask_folder)
    masks.sort()
    for _mask in masks:
        (filename, extension) = os.path.splitext(_mask)
        if extension != '.png':
            continue
        if anno:
            idx = filename + ".jpg"
        else:
            idx = "%05d.jpg" % (int(filename) * 5)
        orig_img = os.path.join(orig_folder, idx)
        mask_img = os.path.join(mask_folder, _mask)
        save_img = os.path.join(save_folder, _mask)
        img_compose(orig_img, mask_img, save_img)
    # "%02d" % i
    return True


if __name__ == "__main__":
    # IMAGE_SAVE_PATH = '/home/ubuntu/code/fengda/MaskTrackRCNN/results/test/flow/compose/0/'
    # if not os.path.exists(IMAGE_SAVE_PATH):
    #     os.makedirs(IMAGE_SAVE_PATH)
    path = "/home/ubuntu/datasets/YT-VIS/results/valid-flow/"
    file_names = os.listdir(path)
    file_names.sort()
    # file_names = ['0065b171f9', '01c76f0a82', '4083cfbe15']
    # import pandas as pd
    # file_names = list(pd.read_csv('val.csv').video_name)
    from tqdm import tqdm
    for file in tqdm(file_names):
        # print(file)
        save_folder = os.path.join("/home/ubuntu/datasets/YT-VIS/results/compose/valid-flow2/", file)
        orig_folder = os.path.join("/home/ubuntu/datasets/YT-VIS/valid/JPEGImages/", file)
        mask_folder = os.path.join("/home/ubuntu/datasets/YT-VIS/results/valid-flow2/", file)
        folder_compose(mask_folder, orig_folder, save_folder, anno=True)

    # orig_folder = "/home/ubuntu/datasets/YT-VIS/train/JPEGImages/05a0a513df"
    # mask_folder = "/home/ubuntu/datasets/YT-VIS/results/train/05a0a513df"
    # save_folder = "/home/ubuntu/datasets/YT-VIS/results/compose/train/05a0a513df"
    # folder_compose(mask_folder, orig_folder, save_folder, anno=True)

    IMAGE_COLUMN = 2
    IMAGE_ROW = 2
    IMAGE_SIZE = (640, 360)
    IMAGE_SAVE_PATH = '/home/ubuntu/code/fengda/MaskTrackRCNN/results/test/compose/'
    if not os.path.exists(IMAGE_SAVE_PATH):
        os.makedirs(IMAGE_SAVE_PATH)

    img1_path = '/home/ubuntu/code/fengda/MaskTrackRCNN/results/test/full/'
    img2_path = '/home/ubuntu/code/fengda/MaskTrackRCNN/results/test/full/'
    img3_path = '/home/ubuntu/code/fengda/MaskTrackRCNN/results/test/full/'

    image_compose()  # 调用函数
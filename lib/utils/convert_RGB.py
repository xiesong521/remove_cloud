#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import numpy as np
from PIL import Image

def convert(origin_img):
    """
    in order to visualization the result
    convert 9_bands img to rgb
    :param origin_img:
    :return:
    """
    # because we o some data_process before train so we need  restore the data

    origin_data = origin_img.data.cpu().numpy()

    origin_data = np.squeeze(origin_data).transpose([1, 2, 0])

    # origin_data = ((np.squeeze(origin_data).transpose([1, 2, 0])) * 0.5 + 0.5)
    # prevent data overflow
    origin_data = np.maximum(np.minimum(origin_data, 1), 0) * 255

    [m, n, p] = origin_data.shape

    RGB = np.zeros([m, n, 3])
    RGB[:, :, 0] = origin_data[:, :, 3]
    RGB[:, :, 1] = origin_data[:, :, 2]
    RGB[:, :, 2] = origin_data[:, :, 1]

    rgb = Image.fromarray(RGB.astype(np.uint8),mode='RGB')
    return rgb



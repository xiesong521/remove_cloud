#!/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""
from skimage.measure import compare_psnr
import math
import numpy as np

# def psnr(img, gt):
#     '''
#
#     :param img:
#     :param gt:
#     :return:
#     '''
#     mse = np.mean((img - gt) ** 2)
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr(img,gt):
    return compare_psnr(gt,img)














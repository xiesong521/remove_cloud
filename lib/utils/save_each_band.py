#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import numpy as np
import os
from PIL import Image
def save_each_band(args, origin_img, name):

    origin_data = origin_img.data.cpu().numpy()
    origin_data = (np.squeeze(origin_data).transpose([1, 2, 0]))

    # origin_data = ((np.squeeze(origin_data).transpose([1, 2, 0])) * 0.5 + 0.5)
    # prevent data overflow
    origin_data = np.maximum(np.minimum(origin_data, 1), 0) * 255

    [m, n, p] = origin_data.shape

    for k in range(p):
        if k == 7 or k == 8:  # because you only save 9 band not sequence
            save_result = args.evaluation_dir + 'B{}/'.format(k + 3)
            os.makedirs(save_result, exist_ok=True)
            # transform array to image
            result_band = Image.fromarray(origin_data[:, :, k].astype(np.uint8), mode='L')
            result_band.save(save_result + '{}.bmp'.format(name))

        else:
            save_result = args.evaluation_dir + 'B{}/'.format(k + 1)
            os.makedirs(save_result, exist_ok=True)
            # transform array to image
            result_band = Image.fromarray(origin_data[:, :, k].astype(np.uint8), mode='L')

            result_band.save(save_result + '{}.bmp'.format(name))

    save_RGB = args.evaluation_dir + 'RGB/'
    os.makedirs(save_RGB, exist_ok=True)

    result_RGB = np.zeros([m, n, 3])

    result_RGB[:, :, 0] = origin_data[:, :, 3]
    result_RGB[:, :, 1] = origin_data[:, :, 2]
    result_RGB[:, :, 2] = origin_data[:, :, 1]

    img_result = Image.fromarray(result_RGB.astype(np.uint8), mode='RGB')

    img_result.save(save_RGB + '{}_r.bmp'.format(name))






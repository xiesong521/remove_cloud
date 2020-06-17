#!/user/bin/python3
# coding:utf-8
"""
    Author:XieSong
    Email:18406508513@163.com

    Copyright:XieSong
    Licence:MIT

"""
import os
from PIL import Image


def merge_channels(rgb_dir, dest_dir):
    path_row = '232086'
    r = Image.open('E:\landsat_dataset\dataset_origin\\' + path_row + '\\free\B4.tif')
    g = Image.open('E:\landsat_dataset\dataset_origin\\' + path_row + '\\free\B3.tif')
    b = Image.open('E:\landsat_dataset\dataset_origin\\' + path_row + '\\free\B2.tif')
    merge = Image.merge('RGB',(r,g,b))
    merge.save('E:\landsat_dataset\dataset_origin\\' + path_row + '\\free\RGB.tif')

    r = Image.open('E:\landsat_dataset\dataset_origin\\' + path_row + '\\cloud\B4.tif')
    g = Image.open('E:\landsat_dataset\dataset_origin\\' + path_row + '\\cloud\B3.tif')
    b = Image.open('E:\landsat_dataset\dataset_origin\\' + path_row + '\\cloud\B2.tif')
    merge = Image.merge('RGB',(r,g,b))
    merge.save('E:\landsat_dataset\dataset_origin\\' + path_row + '\\cloud\RGB.tif')



if __name__ == '__main__':
    rgb_dir = 'C:\\Users\\xs\Desktop\师兄数据集_跑的结果\传统_408\\HF\RGB'
    dest_dir = 'C:\\Users\\xs\Desktop\师兄数据集_跑的结果\传统_408\\HF\\'

    merge_channels(rgb_dir, dest_dir)
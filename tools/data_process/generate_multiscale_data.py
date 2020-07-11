#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import os
import cv2

def generate_path(dest_dir,scale):
    """

    :param dest_dir:
    :param scale:
    :return:
    """
    os.makedirs(os.path.join(dest_dir, str(scale), 'RGB'), exist_ok=True)
    for i in range(1,12):
        if i != 8:
            os.makedirs(os.path.join(dest_dir,str(scale),'B{}'.format(i)),exist_ok=True)

def generate_scale_data(src_dir,dest_dir,scale):
    """

    :param src_dir:
    :param dest_dir:
    :param scale:
    :return:
    """
    list_dir = os.listdir(src_dir)
    for dir in list_dir:
        file_list = os.listdir(os.path.join(src_dir,dir))
        for file in file_list:
            if dir == 'RGB':
                img = cv2.imread(os.path.join(src_dir,dir,file), cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(os.path.join(src_dir,dir,file),cv2.IMREAD_GRAYSCALE)
            size = img.shape
            resize = (int(size[1] * scale), int(size[0] * scale))
            re_img = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(dest_dir,str(scale),dir,file),re_img)


if __name__ == '__main__':

    kind = 'test_cloud/'
    src_dir = '/media/omnisky/data3/xiesong/dataset/cloud/landsat8/' + kind +'1/'
    dest_dir = '/media/omnisky/data3/xiesong/dataset/cloud/landsat8/' + kind
    scale = 0.25
    # scale = 0.5

    generate_path(dest_dir,scale)
    generate_scale_data(src_dir, dest_dir, scale)
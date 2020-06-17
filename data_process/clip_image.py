#!/user/bin/python3
#coding:utf-8
"""
    Author:XieSong
    Email:18406508513@163.com
    
    Copyright:XieSong
    Licence:MIT

"""
from PIL import Image
import os
import numpy as np

def generate_path(dest_dir):
    os.makedirs(os.path.join(dest_dir, 'RGB'), exist_ok=True)
    for i in range(1,12):
        if i != 8:
            os.makedirs(os.path.join(dest_dir,'B{}'.format(i)),exist_ok=True)

def remove_RGB_some_image(root):
    print('remove black image from RGB dir')
    files = os.listdir(root)
    for file in files:
        img = Image.open(os.path.join(root,file))
        img_array = np.array(img)
        min_value = img_array.min()
        if min_value == 0:
            os.remove(os.path.join(root,file))

def find_img_via_RGB(remove_dir,root):
    bands = os.listdir(root)
    file_list = os.listdir(remove_dir)
    for band in bands:
        print("remove black imgs from {}".format(band))
        files = os.listdir(os.path.join(root, band))
        for file in files:
            if file not in file_list:
                os.remove(os.path.join(root,band,file))


def clip(src_dir,dest_dir,clip_w,clip_h):
    dest_path_row = src_dir.split('/')[-3]
    dest_kind = src_dir.split('/')[-2]
    dest_dir = os.path.join(dest_dir, dest_path_row,dest_kind )
    generate_path(dest_dir)

    files = [file for file in os.listdir(src_dir) if file.split('.')[-1] == 'tif']
    for file in files:
        band = file.split('_')[-1].split('.')[0]
        if band != 'B8':
            img = Image.open(os.path.join(src_dir,file))
            w, h = img.size
            print("begin clip picture {}".format(file))
            row_num = h // clip_h
            col_num = w // clip_w
            num = 0
            for r in range(row_num):
                for c in range(col_num):
                    box = (c * clip_w, r * clip_h, (c+1) * clip_w, (r+1) * clip_h)
                    target_img = img.crop(box)
                    target_img.save(os.path.join(dest_dir, band, str(num)+'.bmp'))
                    num += 1
            print('{} clip end with {} small image'.format(file, num))


if __name__ == '__main__':
    src_dir = 'E:/landsat_dataset/dataset_roi/'
    dest_dir = 'E:/landsat_dataset/datasat_clip/'
    path_row = '125033/'
    cloud_kind = 'cloud/'
    free_kind = 'free/'


    remove_dir = dest_dir + path_row + cloud_kind + 'RGB/'

    # src_dir_cloud = 'E:/landsat_dataset/dataset_roi/'+ path_row + cloud_kind
    # clip(src_dir_cloud,dest_dir, 256, 256)
    # print("remove some black image from {}".format(remove_dir))
    # remove_RGB_some_image(remove_dir)
    # src_dir_free = 'E:/landsat_dataset/dataset_roi/'+ path_row + free_kind
    # clip(src_dir_free,dest_dir, 256, 256)

    find_img_via_RGB(remove_dir,dest_dir + path_row + cloud_kind)
    find_img_via_RGB(remove_dir,dest_dir + path_row + free_kind)






#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""

import os


def gen_single_txt(h5_dir,txt_dir,kind):

    os.makedirs(txt_dir,exist_ok=True)
    txt_name = (txt_dir + '{}.txt').format(kind)

    files_list = os.listdir(h5_dir)
    f = open(txt_name, 'w')
    for i in range(len(files_list)):
        filename = files_list[i]
        info = h5_dir + filename
        f.write(info + '\n')
    f.close()


def gen_multi_txt(h5_dir, txt_dir,kind):
    os.makedirs(txt_dir,exist_ok=True)
    txt_name = (txt_dir + '{}.txt').format(kind)
    h5_1 = h5_dir + '1/'
    h5_2 = h5_dir + '0.5/'
    h5_3 = h5_dir + '0.25/'
    files_list = os.listdir(h5_1)
    f = open(txt_name, 'w')
    for i in range(len(files_list)):
        filename = files_list[i]
        info = h5_1 + filename + ' ' + h5_2 + filename + ' ' + h5_3 + filename
        f.write(info + '\n')
    f.close()

if __name__ == '__main__':


    # h5_dir = '/media/omnisky/data3/xiesong/dataset/cloud/landsat8_h5/test/1/'
    # txt_dir = '/media/omnisky/data3/xiesong/datalist/cloud/landsat8/'
    # kind = 'test'
    # gen_single_txt(h5_dir, txt_dir, kind)
    #
    h5_dir = '/media/omnisky/data3/xiesong/dataset/cloud/landsat8_multi_h5/test/'
    txt_dir = '/media/omnisky/data3/xiesong/datalist/cloud/landsat8_multi/'
    kind = 'test'
    gen_multi_txt(h5_dir, txt_dir,kind)

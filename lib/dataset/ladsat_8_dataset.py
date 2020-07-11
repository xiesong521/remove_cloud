#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import os
from PIL import  Image
import random
class Landsat(Dataset):
    def __init__(self,txt_path):
        with open(txt_path,'r') as f:
            self.dir_list=f.readlines()
        self.length=len(self.dir_list)

    def __len__(self):
        return  self.length

    def __getitem__(self, index):
        multi_dir = self.dir_list[index].strip('\n')
        f = h5py.File(multi_dir, 'r')
        name = multi_dir.split('/')[-1].split('.')[0]

        img_cloud = f['data'][:]
        img_free= f['label'][:]

        img=img_cloud.transpose([2,0,1]).astype(np.float32)/255
        gt=img_free.transpose([2,0,1]).astype(np.float32)/255

        # img=(img_cloud.transpose([2,0,1]).astype(np.float32)/255-0.5)/0.5
        # gt=(img_free.transpose([2,0,1]).astype(np.float32)/255-0.5)/0.5
        img = torch.FloatTensor(img)
        gt = torch.FloatTensor(gt)
        return (img,gt,name)


class Landsat_multi_scale(Dataset):
    def __init__(self,txt_path):
        with open(txt_path,'r') as f:
            self.dir_list=f.readlines()
        self.length=len(self.dir_list)

    def __len__(self):
        return  self.length

    def __getitem__(self, index):
        multi_dir = self.dir_list[index].strip('\n').split(' ')

        h5_0 = multi_dir[0]
        h5_1 = multi_dir[1]
        h5_2 = multi_dir[2]

        name = h5_0.split('/')[-1].split('.')[0]

        f0 = h5py.File(h5_0, 'r')
        img_0 = f0['data'][:]
        gt_0= f0['label'][:]

        f1 = h5py.File(h5_1, 'r')
        img_1 = f1['data'][:]
        gt_1 = f1['label'][:]

        f2 = h5py.File(h5_2, 'r')
        img_2 = f2['data'][:]
        gt_2 = f2['label'][:]

        img_0_T=img_0.transpose([2,0,1]).astype(np.float32)/255
        gt_0_T=gt_0.transpose([2,0,1]).astype(np.float32)/255

        img_1_T=img_1.transpose([2,0,1]).astype(np.float32)/255
        gt_1_T=gt_1.transpose([2,0,1]).astype(np.float32)/255


        img_2_T=img_2.transpose([2,0,1]).astype(np.float32)/255
        gt_2_T=gt_2.transpose([2,0,1]).astype(np.float32)/255

        img_0_T = torch.FloatTensor(img_0_T)
        gt_0_T = torch.FloatTensor(gt_0_T)
        img_1_T = torch.FloatTensor(img_1_T)
        gt_1_T = torch.FloatTensor(gt_1_T)
        img_2_T = torch.FloatTensor(img_2_T)
        gt_2_T = torch.FloatTensor(gt_2_T)

        return (img_0_T,gt_0_T,img_1_T,gt_1_T,img_2_T,gt_2_T,name)


class Landsat_RGB(Dataset):
    def __init__(self, args, img_dir,gt_dir, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.args = args
        self.unaligned = unaligned
        self.files_X = sorted(glob.glob(os.path.join(img_dir) + '/*.*'))
        self.files_Y = sorted(glob.glob(os.path.join(gt_dir) + '/*.*'))
        print (len(self.files_X))

    def __getitem__(self, index):

        img_X = Image.open(self.files_X[index % len(self.files_X)])
        if self.unaligned:
            img_Y = Image.open(self.files_Y[random.randint(0, len(self.files_Y)-1)])
        else:
            img_Y = Image.open(self.files_Y[index % len(self.files_Y)] )

        img_X = self.transform(img_X)
        img_Y = self.transform(img_Y)

        # if self.args.input_nc_A == 1:  # RGB to gray
        #     img_X = img_X.convert('L')
        #
        # if self.args.input_nc_B == 1:  # RGB to gray
        #     img_Y = img_Y.convert('L')

        return {'X': img_X, 'Y': img_Y}

    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))








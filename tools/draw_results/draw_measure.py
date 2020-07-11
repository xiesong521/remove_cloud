#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
'draw psnr and ssim loss in tensorboard to select the best model'
import os
from RSC.ssim import get_mean_ssim
from RSC.psnr import get_mean_psnr
from RSC.options import TrainOptions
from tensorboardX import SummaryWriter

args = TrainOptions().parse()
writer = SummaryWriter(comment='scalar11')
def get_best_model(img_dir):
    result_list = os.listdir(img_dir)
    result_list = sorted(result_list, key=lambda x: os.path.getmtime(os.path.join(img_dir, x)))
    print(result_list)

    for i in range(len(result_list)):
        result_path = img_dir + str(result_list[i]) +'/result/'
        print(result_path)
        target_path = '/home/data3/xs077/Code/pix2pix_rm_cloud/pix2pix-pytorch-master/Exp4/552/target/'
        ssim_ = get_mean_ssim(result_path, target_path)
        writer.add_scalar('scalar/ssim', ssim_,i)
        psnr_ = get_mean_psnr(result_path, target_path)
        writer.add_scalar('scalar/psnr', psnr_,i)
    writer.close()

img_dir = '/home/data3/xs077/Code/RSC/RSC11-mydata2_h5/every_model/'
get_best_model(img_dir)


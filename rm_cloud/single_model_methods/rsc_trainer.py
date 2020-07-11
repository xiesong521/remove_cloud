#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
from rm_cloud.single_model_methods.trainer import Trainer
from lib.loss.mse_loss import get_mse_loss_function
import torch.nn as nn

class RscTrainer(Trainer):
    '''
    paper:[2019_ISPRS] Thin cloud removal with residual symmetrical concatenation network
    link:https://kopernio.com/viewer?doi=10.1016/j.isprsjprs.2019.05.003&route=6
    '''
    def __init__(self,model,optimizer):
        super().__init__(model,optimizer)

    def optimize_strategy(self,img,gt):
        gen_imgs = self.model(img)
        mse_loss = get_mse_loss_function()
        loss = mse_loss(gen_imgs,gt)

        #BP
        self.optimizer.zero_grad()
        loss.backward()
        #jinxing tidu caijian  yinwei lossNaN
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()

        return gen_imgs,loss



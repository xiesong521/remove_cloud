#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
from rm_cloud.gan_methods.trainer import Trainer

class McganTrainer(Trainer):

    def __init__(self,dis, gen , D_optimizer, G_optimizer):
        super().__init__(dis, gen , D_optimizer, G_optimizer)

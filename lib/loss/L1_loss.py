#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import torch

def get_l1_loss_function():
    """

    :return: pixelwise_loss
    """
    l1_loss = torch.nn.L1Loss()
    if torch.cuda.is_available():
        l1_loss.cuda()
    return l1_loss
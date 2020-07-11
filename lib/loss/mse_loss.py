#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import torch

def get_mse_loss_function():
    """

    :return: pixelwise_loss
    """
    mse_loss = torch.nn.MSELoss()
    if torch.cuda.is_available():
        mse_loss.cuda()
    return mse_loss
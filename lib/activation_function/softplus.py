#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import torch
import torch.nn as nn

def get_softplus():

    if torch.cuda.is_available():
        softplus = nn.Softplus().cuda()
    else:
        softplus = nn.Softplus()
    return softplus


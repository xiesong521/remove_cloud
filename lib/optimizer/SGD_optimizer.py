#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import torch

def get_sgd_optimizer(args, model):
    opimizer = torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=1e-4)
    return opimizer
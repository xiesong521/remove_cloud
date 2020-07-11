#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import torch
import itertools

def get_adam_optimizer(args, model):
    opimizer = torch.optim.Adam(model.parameters(),lr=args.lr, betas=(args.b1, args.b2),weight_decay=1e-4)
    return opimizer

def get_adam_optimizer_chain(args, model1, model2):
    opimizer = torch.optim.Adam(itertools.chain(model1.parameters(),model2.parameters()),lr=args.lr, betas=(args.b1, args.b2),weight_decay=1e-4)
    return opimizer
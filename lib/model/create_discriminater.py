#!/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""
import torch
import torch.nn as nn
import math


def print_network(model):
    """

    :param model:
    :return: print network structure
    """
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(model)
    print('Total number of parameters: %d' % num_params)


def weight_init_mean(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def create_discriminator(args, discriminator):
    print_network(discriminator)
    if torch.cuda.is_available():
        discriminator = discriminator.cuda()
    if args.mGPU:
        discriminator = nn.DataParallel(discriminator, args.device_ids)

    weight_init_mean(discriminator)

    if args.start_epoch != 0:
        discriminator.load_state_dict(torch.load(args.checkpoint_dis))

    return discriminator
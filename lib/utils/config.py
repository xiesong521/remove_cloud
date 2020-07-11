#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""

import argparse

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--config_file', type=str, default='/home/data1/xiesong/git_repo/removal_thin_cloud/experiments/ssa/config.yaml', help='yaml config file')
        self.parser.add_argument('--devices', type=str, default='4,5', help='gpu')
        self.parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
        self.parser.add_argument('--channels', type=int, default=1, help='number of image channels')
        self.parser.add_argument('--start_epoch', type=int, default=713, help='epoch to start training from')
        self.parser.add_argument('--end_epoch', type=int, default=20000, help='epoch to end training from')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='adam:learning rate')
        self.parser.add_argument('--b1', type=float, default=0.9, help='decay of first order momentum of gradient')
        self.parser.add_argument('--b2', type=float, default=0.999, help='decay of second order momentum of gradient')
        self.parser.add_argument('--batch_size', type=int, default=64, help='size of batches')
        self.parser.add_argument('--sample_interval', type=int, default=10000, help='interval between image samples')
        self.parser.add_argument('--mGPU', default=True, help='the mutiple cuda to use')



    def parse(self):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()
        self.args = args
        return self.args

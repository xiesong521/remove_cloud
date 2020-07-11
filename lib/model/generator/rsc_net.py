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
import torch.nn.functional as F


#####################
######RSC_Net########
#####################
class Block(nn.Module):
    def __init__(self, channels):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.deconv1 = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv2 = nn.ConvTranspose2d(2*channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv3 = nn.ConvTranspose2d(2*channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv4 = nn.ConvTranspose2d(2*channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv5 = nn.ConvTranspose2d(2*channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, input):

        c_1 = self.conv1(input)
        residual1 = F.relu(torch.add(input, c_1))

        c_2 = self.conv2(residual1)
        residual2 = F.relu(torch.add(residual1, c_2))

        c_3 = self.conv3(residual2)
        residual3 = F.relu(torch.add(residual2, c_3))

        c_4 = self.conv4(residual3)
        residual4 = F.relu(torch.add(residual3, c_4))

        c_5 = self.conv5(residual4)
        residual5 = F.relu(torch.add(residual4, c_5))


        d_1 = self.deconv1(residual5)
        residual6 = F.relu(torch.add(residual5, d_1))

        com1 = torch.cat((residual6, residual5), 1)

        d_2 = self.deconv2(com1)
        residual7 = F.relu(torch.add(residual6, d_2))

        com2 = torch.cat((residual7, residual4), 1)

        d_3 = self.deconv3(com2)
        residual8 = F.relu(torch.add(residual7, d_3))

        com3 = torch.cat((residual8, residual3), 1)

        d_4 = self.deconv4(com3)
        residual9 = F.relu(torch.add(residual8, d_4))

        com4 = torch.cat((residual9, residual2), 1)

        d_5 = self.deconv5(com4)
        residual10 = F.relu(torch.add(residual9, d_5))

        com5 = torch.cat((residual10, residual1), 1)

        return com5


class RSC_Net(nn.Module):
    def __init__(self):
        super(RSC_Net, self).__init__()
        self.input = nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.block = Block(64)
        self.output = nn.ConvTranspose2d(128, 9, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        feature = self.input(x)
        c = self.block(feature)
        y = self.output(c)
        return y






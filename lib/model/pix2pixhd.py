#!/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


def conv5x5(inplanes, outplanes, stride=1):
    ' 5x5 convolution with padding'
    return nn.Conv2d(inplanes, outplanes, kernel_size=5, stride=stride, padding=2, bias=True)


def conv3x3(inplanes, outplanes, stride=1):
    ' 3x3 convolution with padding'
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv1x1(inplanes, outplanes, stride=1):
    ' 1x1 convolution without padding'
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=True)


def deconv3x3(inplanes, outplanes, stride=1):
    ' 3x3 deconvolution with padding'
    return nn.ConvTranspose2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=True)


def deconv2x2(inplanes, outplanes, stride=2):
    ' 2x2 deconvolution without padding'
    return nn.ConvTranspose2d(inplanes, outplanes, kernel_size=2, stride=stride, padding=0, bias=True)


def deconv1x1(inplanes, outplanes, stride=1):
    ' 1x1 deconvolution without padding'
    return nn.ConvTranspose2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=True)


def residual_conv(inplanes, outplanes, stride=1):
    conv_block = nn.Sequential(
        nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=True),
    )
    return conv_block


def residual_deconv(inplanes, outplanes, stride=1):
    conv_block = nn.Sequential(
        nn.ConvTranspose2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.ReLU(),
        nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=True),
    )
    return conv_block


class Block_skip(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = residual_conv(channels, channels)
        self.conv2 = residual_conv(channels, channels)
        self.conv3 = residual_conv(channels, channels)
        self.conv4 = residual_conv(channels, channels)
        self.conv5 = residual_conv(channels, channels)
        self.map = conv1x1(channels, channels)

        self.deconv1 = residual_deconv(channels, channels)
        self.deconv2 = residual_deconv(2 * channels,channels)
        self.deconv3 = residual_deconv(2 * channels,channels)
        self.deconv4 = residual_deconv(2 * channels,channels)
        self.deconv5 = residual_deconv(2 * channels,channels)

    def forward(self, x):
        l_1 = self.conv1(x)
        residual1 = F.relu(torch.add(x, l_1))

        l_2 = self.conv2(residual1)
        residual2 = F.relu(torch.add(residual1, l_2))

        l_3 = self.conv3(residual2)
        residual3 = F.relu(torch.add(residual2, l_3))

        l_4 = self.conv4(residual3)
        residual4 = F.relu(torch.add(residual3, l_4))

        l_5 = self.conv5(residual4)
        residual5 = F.relu(torch.add(residual4, l_5))

        map = F.relu(self.map(residual5))

        r_1 = self.deconv1(map)
        residual6 = F.relu(torch.add(map, r_1))
        com1 = torch.cat((residual6, residual5), 1)

        r_2 = self.deconv2(com1)
        residual7 = F.relu(torch.add(residual6, r_2))
        com2 = torch.cat((residual7, residual4), 1)

        r_3 = self.deconv3(com2)
        residual8 = F.relu(torch.add(residual7, r_3))
        com3 = torch.cat((residual8, residual3), 1)

        r_4 = self.deconv4(com3)
        residual9 = F.relu(torch.add(residual8, r_4))
        com4 = torch.cat((residual9, residual2), 1)

        r_5 = self.deconv5(com4)
        residual10 = F.relu(torch.add(residual9, r_5))
        com5 = torch.cat((residual10, residual1), 1)
        return com5


class SS_Net(nn.Module):
    def __init__(self):
        super(SS_Net, self).__init__()
        self.input = conv3x3(9, 16)
        self.block = Block_skip(16)
        self.output = deconv3x3(32, 9)


    def forward(self, x):
        feature = self.input(x)
        l = self.block(feature)
        o = self.output(l)
        return o




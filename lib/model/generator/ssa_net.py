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


# def residual_deconv(inplanes, outplanes, stride=1):
#     conv_block = nn.Sequential(
#         nn.ConvTranspose2d(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=True),
#         nn.ReLU(),
#         nn.ConvTranspose2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=True),
#     )
#     return conv_block

def residual_deconv(inplanes, outplanes, stride=1):
    conv_block = nn.Sequential(
        nn.ConvTranspose2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.ReLU(),
        nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=True),
    )
    return conv_block


class skip_attention(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(skip_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.convhigh = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=0, bias=True)
        self.convlow = nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, high, low):
        b, c, _, _ = high.size()
        w = self.avg_pool(high).view(b, c)#[8,16,256,256]biancheng [8,16,1,1]
        w = self.convhigh(w.view(b, c, 1, 1))
        #w and d is equal buzhidao  weisha yao view duociyiju
        # print(w)
        # d = self.avg_pool(high)
        # d = self.convhigh(d)
        # print(d)
        low = self.convlow(low)
        a = w * low
        return w * low


class Block_skip_attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = residual_conv(channels, channels)
        self.conv2 = residual_conv(channels, channels)
        self.conv3 = residual_conv(channels, channels)
        self.conv4 = residual_conv(channels, channels)
        self.conv5 = residual_conv(channels, channels)
        self.map = conv1x1(channels, channels)

        self.deconv1 = residual_deconv(channels, channels)
        self.skip1 = skip_attention(channels, channels)
        self.deconv2 = residual_deconv(2 * channels,channels)
        self.skip2 = skip_attention(channels, channels)
        self.deconv3 = residual_deconv(2 * channels,channels)
        self.skip3 = skip_attention(channels, channels)
        self.deconv4 = residual_deconv(2 * channels, channels)
        self.skip4 = skip_attention(channels, channels)
        self.deconv5 = residual_deconv(2 * channels, channels)
        self.skip5 = skip_attention(channels, channels)

        # self.deconv1 = residual_deconv(channels, channels)
        # self.skip1 = skip_attention(channels, channels)
        # self.deconv2 = residual_deconv(2 * channels, 2 * channels)
        # self.skip2 = skip_attention(2 *channels, channels)
        # self.deconv3 = residual_deconv(3 * channels,3 * channels)
        # self.skip3 = skip_attention(3 *channels, channels)
        # self.deconv4 = residual_deconv(4 * channels, 4 *channels)
        # self.skip4 = skip_attention(4 * channels, channels)
        # self.deconv5 = residual_deconv(5 * channels, 5 * channels)
        # self.skip5 = skip_attention(5 * channels,channels)

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
        skip1 = self.skip1(residual6, residual5)
        com1 = torch.cat((residual6, skip1), 1)

        r_2 = self.deconv2(com1)
        residual7 = F.relu(torch.add(residual6, r_2))
        # residual7 = F.relu(torch.add(com1, r_2))
        skip2 = self.skip2(residual7, residual4)
        com2 = torch.cat((residual7, skip2), 1)

        r_3 = self.deconv3(com2)
        residual8 = F.relu(torch.add(residual7, r_3))
        # residual8 = F.relu(torch.add(com2, r_3))
        skip3 = self.skip3(residual8, residual3)
        com3 = torch.cat((residual8, skip3), 1)

        r_4 = self.deconv4(com3)
        residual9 = F.relu(torch.add(residual8, r_4))
        # residual9 = F.relu(torch.add(com3, r_4))
        skip4 = self.skip4(residual9, residual2)
        com4 = torch.cat((residual9, skip4), 1)

        r_5 = self.deconv5(com4)
        residual10 = F.relu(torch.add(residual9, r_5))
        # residual10 = F.relu(torch.add(com4, r_5))
        skip5 = self.skip5(residual10, residual1)
        com5 = torch.cat((residual10, skip5), 1)

        return com5


class SSA_Net(nn.Module):
    def __init__(self):
        super(SSA_Net, self).__init__()
        self.input = conv3x3(9, 16)  # 1
        self.block = Block_skip_attention(16)
        self.output = deconv3x3(32, 9)
        # self.output = deconv3x3(6 * 16, 9)


    def forward(self, x):
        feature = self.input(x)
        l = self.block(feature)
        o = self.output(l)
        return o




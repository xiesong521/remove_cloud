#!/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""
from rm_cloud.single_model_methods.trainer import Trainer
from lib.loss.mse_loss import get_mse_loss_function


class SsTrainer(Trainer):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    def optimize_strategy(self, img, gt):
        gen_imgs = self.model(img)
        mse_loss = get_mse_loss_function()
        loss = mse_loss(gen_imgs, gt)

        # BP
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return gen_imgs, loss


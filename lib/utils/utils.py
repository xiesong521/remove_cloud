#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import os
import cv2
import random

import torch
from torch.backends import cudnn
from torch.autograd import Variable


def gpu_manage(config):
    if config.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.gpu_ids))
        config.gpu_ids = list(range(len(config.gpu_ids)))

    print(os.environ['CUDA_VISIBLE_DEVICES'])

    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    print('Random Seed: ', config.manualSeed)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not config.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def save_image(out_dir, x, num, epoch, filename=None):
    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path = os.path.join(test_dir, filename)
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.png'.format(num))

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    cv2.imwrite(test_path, x)


def checkpoint(config, epoch, gen, dis):
    model_dir = os.path.join(config.out_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    net_gen_model_out_path = os.path.join(model_dir, 'gen_model_epoch_{}.pth'.format(epoch))
    net_dis_model_out_path = os.path.join(model_dir, 'dis_model_epoch_{}.pth'.format(epoch))
    torch.save(gen.state_dict(), net_gen_model_out_path)
    torch.save(dis.state_dict(), net_dis_model_out_path)
    print("Checkpoint saved to {}".format(model_dir))


def make_manager():
    if not os.path.exists('.job'):
        os.makedirs('.job')
        with open('.job/job.txt', 'w') as f:
            f.write('0')


def job_increment():
    with open('.job/job.txt', 'r') as f:
        n_job = f.read()
        n_job = int(n_job)
    with open('.job/job.txt', 'w') as f:
        f.write(str(n_job + 1))

    return n_job


class ReplayBuffer():

    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, epoch_num, epoch_start, decay_start_epoch):
        assert ((epoch_num - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.epoch_num = epoch_num
        self.epoch_start = epoch_start
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + 1 + self.epoch_start - self.decay_start_epoch)/(self.epoch_num - self.decay_start_epoch)


#!/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""
import os

import torch

from lib.dataset.landsat_8_dataloader import Get_dataloader
from lib.model.create_generator import create_generator
from lib.model.generator.ms_net import MS_Net
from lib.model.generator.msa_net import MSA_Net
from lib.model.generator.rsc_net import RSC_Net
from lib.model.generator.ssa_net import SSA_Net
from lib.model.generator.ss_net import SS_Net



from lib.utils.config import Config
from lib.utils.parse_args import parse_args
from rm_cloud.single_model_methods.tester import Testner


def test():
    config = Config().parse()
    args = parse_args(config)
    print(args)
    train_loader, test_loader = Get_dataloader(args)
    checkpoint = args.checkpoint
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    rsc = RSC_Net()
    msa = MSA_Net()
    ssa = SSA_Net()
    ms = MS_Net()
    ss = SS_Net()

    if args.model.arch == 'rsc':
        model = create_generator(args, rsc)
        model.load_state_dict(torch.load(checkpoint))
        rsc = Testner(model)
        rsc.test(args, test_loader)
        rsc.eval(args)

    if args.model.arch == 'msa':
        model = create_generator(args, msa)
        model.load_state_dict(torch.load(checkpoint))
        msa = Testner(model)
        msa.test(args, test_loader)
        msa.eval(args)

    if args.model.arch == 'ss':
        model = create_generator(args, ss)
        model.load_state_dict(torch.load(checkpoint))
        ss = Testner(model)
        ss.test(args, test_loader)
        ss.eval(args)



    if args.model.arch == 'ssa':
        model = create_generator(args, ssa)
        model.load_state_dict(torch.load(checkpoint))
        msa = Testner(model)
        msa.test(args, test_loader)
        msa.eval(args)

    if args.model.arch == 'ms':
        model = create_generator(args, ms)
        model.load_state_dict(torch.load(checkpoint))
        ms = Testner(model)
        ms.test(args, test_loader)
        ms.eval(args)
    if args.model.arch == 'msa_':
        model = create_generator(args, msa)
        model.load_state_dict(torch.load(checkpoint))
        msa_ = Testner(model)
        msa_.test(args, test_loader)
        msa_.eval(args)




if __name__ == '__main__':

    test()
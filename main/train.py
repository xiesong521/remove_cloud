#!/usr/bin/python3
# coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com

    Copyright: Xie Song
    License: MIT
"""
import os

from lib.dataset.landsat_8_dataloader import Get_dataloader
from lib.dataset.landsat_8_dataloader import Get_dataloader_RGB


from lib.model.create_generator import create_generator
from lib.model.create_discriminater import create_discriminator
from lib.model.discriminator.mcGAN_dis import Discriminator
from lib.model.generator.mcGAN_gen import UNet
from lib.model.generator.ms_net import MS_Net
from lib.model.generator.msa_net import MSA_Net
from lib.model.generator.rsc_net import RSC_Net
from lib.model.generator.ssa_net import SSA_Net
from lib.model.generator.ss_net import SS_Net

from lib.model.generator.cloudGan_gen import GeneratorResNet
from lib.model.discriminator.cloudGan_dis import Discriminator_n_layers
from lib.optimizer.Adam_optimizer import get_adam_optimizer
from lib.optimizer.Adam_optimizer import get_adam_optimizer_chain

from lib.optimizer.SGD_optimizer import get_sgd_optimizer
from lib.utils.config import Config
from lib.utils.parse_args import parse_args
from rm_cloud.gan_methods.mcgan_trainer import McganTrainer
from rm_cloud.gan_methods.cloudGan_trainer import CloudGanTrainer
from rm_cloud.single_model_methods.ms_trainer import MsTrainer
from rm_cloud.single_model_methods.msa__trainer import Msa_Trainer
from rm_cloud.single_model_methods.msa_trainer import MsaTrainer
from rm_cloud.single_model_methods.rsc_trainer import RscTrainer
from rm_cloud.single_model_methods.ssa_trainer import SsaTrainer
from rm_cloud.single_model_methods.ss_trainer import SsTrainer


def train():
    config = Config().parse()
    args = parse_args(config)
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    if args.in_channels == 9:
        train_loader,test_loader = Get_dataloader(args)
    else:
        train_loader, test_loader = Get_dataloader_RGB(args)

    if args.model.arch == 'rsc':
        rsc = RSC_Net()
        model = create_generator(args, rsc)
        # adm_optimizer = get_adam_optimizer(args, model)
        # rsc = RscTrainer(model, adm_optimizer)
        sgd_optimizer = get_sgd_optimizer(args, model)
        rsc = RscTrainer(model, sgd_optimizer)
        rsc.train_single(args, train_loader, args.start_epoch, args.end_epoch)

    if args.model.arch == 'ss':
        ss = SS_Net()
        model = create_generator(args, ss)
        # sgd_optimizer = get_sgd_optimizer(args, model)
        # ss = SsTrainer(model, sgd_optimizer)
        adm_optimizer = get_adam_optimizer(args, model)
        ss = RscTrainer(model, adm_optimizer)
        ss.train_single(args, train_loader, args.start_epoch, args.end_epoch)

    # if args.model.arch == 'ss+':
    #     ss = SS_Net()
    #     model = create_generator(args, ss)
    #     sgd_optimizer = get_sgd_optimizer(args, model)
    #     ss = SsTrainer(model, sgd_optimizer)
    #     ss.train_single(args, train_loader, args.start_epoch, args.end_epoch)

    if args.model.arch == 'msa':

        msa = MSA_Net()
        model = create_generator(args, msa)
        adm_optimizer = get_adam_optimizer(args, model)
        msa = MsaTrainer(model, adm_optimizer)
        msa.train_multi(args, train_loader, args.start_epoch, args.end_epoch)

    if args.model.arch == 'ssa':
        ssa = SSA_Net()
        model = create_generator(args, ssa)
        adm_optimizer = get_adam_optimizer(args, model)
        ssa = SsaTrainer(model, adm_optimizer)
        ssa.train_single(args, train_loader, args.start_epoch, args.end_epoch)



    if args.model.arch == 'msa_':
        msa = MSA_Net()
        model = create_generator(args, msa)
        adm_optimizer = get_adam_optimizer(args, model)
        ms = Msa_Trainer(model, adm_optimizer)
        ms.train_multi(args, train_loader, args.start_epoch, args.end_epoch)


    if args.model.arch == 'ms':
        ms = MS_Net()
        model = create_generator(args, ms)
        adm_optimizer = get_adam_optimizer(args, model)
        ms = MsTrainer(model, adm_optimizer)
        ms.train_multi(args, train_loader, args.start_epoch, args.end_epoch)

    if args.model.arch == 'mcgan':
        mcgan_gen = UNet(args)
        mcgan_dis = Discriminator(args)
        gen = mcgan_gen
        dis = mcgan_dis
        opt_gen = get_adam_optimizer(args, gen)
        opt_dis = get_adam_optimizer(args, dis)
        mcgan = McganTrainer(gen, dis, opt_dis, opt_gen)
        mcgan.train(args, train_loader,test_loader, args.start_epoch, args.end_epoch)

    if args.model.arch == 'cloudGan':
        cloudGan_G_AB = GeneratorResNet(args)
        cloudGan_gen_AB = create_generator(args,cloudGan_G_AB)

        cloudGan_dis_B = Discriminator_n_layers(args)
        cloudGan_dis_B = create_discriminator(args,cloudGan_dis_B)

        cloudGan_gen_BA = GeneratorResNet(args)
        cloudGan_gen_BA = create_generator(args,cloudGan_gen_BA)

        cloudGan_dis_A = Discriminator_n_layers(args)
        cloudGan_dis_A = create_discriminator(args,cloudGan_dis_A)

        opt_gen = get_adam_optimizer_chain(args,cloudGan_gen_AB,cloudGan_gen_BA)
        opt_dis_A = get_adam_optimizer(args,cloudGan_dis_A)
        opt_dis_B = get_adam_optimizer(args,cloudGan_dis_B)
        cloudGan = CloudGanTrainer(cloudGan_gen_AB,cloudGan_gen_BA,cloudGan_dis_A,cloudGan_dis_B,opt_gen,opt_dis_A,opt_dis_B)
        cloudGan.train(args, train_loader[0],test_loader[0], args.start_epoch, args.end_epoch)




if __name__ == '__main__':
    train()


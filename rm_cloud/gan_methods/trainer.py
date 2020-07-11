#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import torch
from torch.autograd import Variable
from lib.loss.mse_loss import get_mse_loss_function
from lib.loss.L1_loss import get_l1_loss_function
from lib.activation_function.softplus import get_softplus
from lib.utils.log_report import LogReport,TestReport
from rm_cloud.gan_methods.tester import test
from lib.utils.utils import gpu_manage, save_image, checkpoint

class Trainer(object):
    def __init__(self,dis,gen,D_optimizer,G_optimizer):
        self.dis = dis.train()
        self.gen = gen.train()
        self.D_optimizer = D_optimizer
        self.G_optimizer = G_optimizer


    def train(self,args,train_dataloader,test_dataloader,start_epoch,end_epoch):

        logreport = LogReport(log_dir = args.config.log_dir)
        testreport = TestReport(log_dir=args.config.out_dir)

        print("======== begin train model ========")

        for epoch in range(start_epoch,end_epoch):
            for i ,(img,gt,name) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    img = Variable(img.cuda())
                    gt = Variable(gt.cuda())
                else:
                    img = Variable(img)
                    gt = Variable(gt)
                fake_gt = self.gen.forward(img)

                ########################
                ########Update D########
                ########################
                self.D_optimizer.zero_grad()

                # train with fake
                fake_img_gt = torch.cat((img, fake_gt), 1)
                pred_fake = self.dis.forward(fake_img_gt.detach())
                batchsize, _, w, h = pred_fake.size()

                loss_d_fake = torch.sum(get_softplus(-pred_fake)) / batchsize / w / h

                # train with real
                real_img_gt = torch.cat((img, gt), 1)
                pred_real = self.dis.forward(real_img_gt)
                loss_d_real = torch.sum(get_softplus(-pred_real)) / batchsize / w / h

                # combined loss

                loss_d = loss_d_fake + loss_d_real

                loss_d.backward()

                if epoch % args.minimax == 0:
                    self.D_optimizer.step()

                ########################
                ########Update G########
                ########################
                self.G_optimizer.zero_grad()

                # First , G(A) should fake the discriminator
                fake_img_gt = torch.cat((img, fake_gt), 1)
                pred_fake = self.dis.forward(fake_img_gt)
                loss_g_gan = torch.sum(get_softplus(-pred_fake)) / batchsize / w /h

                # Second  G(A) = B

                loss_g_l1 = get_l1_loss_function(fake_gt,gt) * args.config.lamb

                loss_g = loss_g_gan + loss_g_l1

                loss_g.backward()

                self.G_optimizer.step()

                # log
                if i % 100 == 0:
                    print(
                        "===> Epoch[{}]({}/{}): loss_d_fake: {:.4f} loss_d_real: {:.4f} loss_g_gan: {:.4f} loss_g_l1: {:.4f}".format(
                            epoch, i, len(train_dataloader), loss_d_fake.item(), loss_d_real.item(),
                            loss_g_gan.item(), loss_g_l1.item()))

                    log = {}

                    log['epoch'] = epoch
                    log['iteration'] = len(train_dataloader) * (epoch - 1) + i
                    log['gen/loss'] = loss_g.item()
                    log['dis/loss'] = loss_d.item()

                    logreport(log)

                    with torch.no_grad():
                        log_test = test(args, test_dataloader, self.gen, get_mse_loss_function(), epoch)
                        testreport(log_test)

                    if epoch % args.snapshot_interval == 0:
                        checkpoint(args, epoch, self.gen, self.dis)

                    logreport.save_lossgraph()
                    testreport.save_lossgraph()








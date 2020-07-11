#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import os
import numpy as np

import time
from lib.utils.convert_RGB import convert

import torch
from torch.autograd import Variable
from lib.loss.mse_loss import get_mse_loss_function
from lib.loss.L1_loss import get_l1_loss_function
from lib.utils.utils import ReplayBuffer, LambdaLR
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

class CloudGanTrainer(object):
    def __init__(self,gen_AB,gen_BA,dis_A,dis_B,optimizer_G,optimizer_D_A,optimizer_D_B):
        self.gen_AB = gen_AB.train()
        self.gen_BA = gen_BA.train()
        self.dis_A = dis_A.train()
        self.dis_B = dis_B.train()
        self.optimizer_G = optimizer_G
        self.optimizer_D_A = optimizer_D_A
        self.optimizer_D_B = optimizer_D_B

    def train(self,args,train_dataloader,test_dataloader,start_epoch,end_epoch):

        patch = (1,args.img_height//(2**args.n_D_layers*4), args.img_width//(2**args.n_D_layers*4))
        fake_img_buffer = ReplayBuffer()
        fake_gt_buffer = ReplayBuffer()

        writer = SummaryWriter(log_dir='{}'.format(args.log_dir),comment='train_loss')
        print("======== begin train model ========")
        print('data_size:',len(train_dataloader))

        best_loss = 3
        os.makedirs(args.results_gen_model_dir, exist_ok=True)
        os.makedirs(args.results_dis_model_dir, exist_ok=True)
        os.makedirs(args.results_img_dir, exist_ok=True)
        os.makedirs(args.results_gt_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)


        for epoch in range(start_epoch,end_epoch):
            for i,batch in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    img = Variable(batch['X'].type(torch.FloatTensor).cuda())
                    gt = Variable(batch['Y'].type(torch.FloatTensor).cuda())


                else:
                    img = Variable(batch['X'].type(torch.FloatTensor))
                    gt = Variable(batch['Y'].type(torch.FloatTensor))

                valid = Variable(torch.FloatTensor(np.ones((img.size(0), *patch))).cuda(), requires_grad=False)
                fake = Variable(torch.FloatTensor(np.zeros((img.size(0), *patch))).cuda(), requires_grad=False)

                ##### Train Generator #######
                self.optimizer_G.zero_grad()
                # identity loss
                identity_loss = get_l1_loss_function()
                loss_id_img = identity_loss(self.gen_BA(img),img)
                loss_id_gt = identity_loss(self.gen_AB(gt),gt)
                loss_identity = (loss_id_gt + loss_id_img) / 2

                # GAN loss
                fake_gt = self.gen_AB(img)
                pred_fake = self.dis_B(fake_gt)
                gan_loss = get_mse_loss_function()
                loss_GAN_img_gt = gan_loss(pred_fake,valid)

                fake_img = self.gen_BA(gt)
                pred_fake = self.dis_B(fake_img)
                loss_GAN_gt_img = gan_loss(pred_fake, valid)

                loss_GAN = (loss_GAN_img_gt + loss_GAN_gt_img) / 2

                #Cycle loss
                recov_img = self.gen_BA(fake_gt)
                cycle_loss = get_l1_loss_function()
                loss_cycle_img = cycle_loss(recov_img,img)

                recov_gt = self.gen_BA(fake_img)
                loss_cycle_gt = cycle_loss(recov_gt,gt)

                loss_cycle = (loss_cycle_gt + loss_cycle_img) / 2

                # Tota loss
                loss_G = loss_GAN + args.lambda_id * loss_identity + args.lambda_cyc * loss_cycle

                loss_G.backward()
                self.optimizer_G.step()
                batches_done = epoch * len(train_dataloader) + i


                ####### Train Discriminator A #######
                self.optimizer_D_A.zero_grad()
                pred_real = self.dis_A(img)
                loss_real = gan_loss(pred_real,valid)
                fake_img = fake_img_buffer.push_and_pop(fake_img)
                pred_fake = self.dis_A(fake_img.detach())
                loss_fake = gan_loss(pred_fake,fake)

                loss_D_img = (loss_real + loss_fake) / 2
                loss_D_img .backward()
                self.optimizer_D_A.step()

                ####### Train Discriminator B #######
                self.optimizer_D_B.zero_grad()
                pred_real = self.dis_B(gt)
                loss_real = gan_loss(pred_real, valid)
                fake_gt = fake_gt_buffer.push_and_pop(fake_gt)
                pred_fake = self.dis_B(fake_gt.detach())
                loss_fake = gan_loss(pred_fake, fake)

                loss_D_gt = (loss_real + loss_fake) / 2
                loss_D_gt.backward()
                self.optimizer_D_B.step()

                loss_D = (loss_D_img + loss_D_gt) / 2

                writer.add_scalars('{}_train_loss'.format(args.model.arch),{'loss_G':loss_G.data.cpu(),'loss_D':loss_D.data.cpu()}, batches_done)


                f= open(os.path.join(args.log_dir, 'log.txt'),'a+')
                info = 'epoch:' + str(epoch) + ' batches_done:' + str(batches_done) + ' loss_GAN:' + str(loss_GAN.data.cpu())\
                       + ' loss_identity:' + str(loss_identity.data.cpu())+' loss_identity:'+ str(loss_identity) + ' loss_cycle:'\
                       + str(loss_cycle.data.cpu()) + ' loss_G:' + str(loss_G.data.cpu())+ ' loss_D_gt:' + str(loss_D_gt.data.cpu()) + ' loss_D_img:' + str(loss_D_img.data.cpu())
                f.write(info + '\n')

                ########## save best result ##############

                if loss_G.data.cpu() < best_loss:
                    best_loss = loss_G.data.cpu()
                    torch.save(self.gen_AB.state_dict(), args.results_gen_model_dir + '/%d-%d_gen_AB_best_model.pkl'%(epoch,batches_done))
                    torch.save(self.gen_BA.state_dict(), args.results_gen_model_dir + '/%d-%d_gen_BA_best_model.pkl'%(epoch,batches_done))
                    torch.save(self.dis_A.state_dict(), args.results_dis_model_dir + '/%d-%d_dis_A_best_model.pkl'%(epoch,batches_done))
                    torch.save(self.dis_B.state_dict(), args.results_dis_model_dir + '/%d-%d_dis_B_best_model.pkl'%(epoch,batches_done))
                    save_image(fake_gt, '%s/%s-%s.bmp' % (args.results_gt_dir, epoch, batches_done), nrow=4,normalize=True)
                    save_image(fake_img, '%s/%s-%s.bmp' % (args.results_img_dir, epoch, batches_done), nrow=4,normalize=True)

                if i % args.interval == 0:
                    print('[epoch %d/%d] [batch %d/%d] [loss: %f]' %(epoch,end_epoch,batches_done,(end_epoch*len(train_dataloader)),loss_G.item()))

            if epoch % args.interval == 0:
                torch.save(self.gen_AB.state_dict(),
                           args.results_gen_model_dir + '/%d-%d_gen_AB.pkl' % (epoch, batches_done))
                torch.save(self.gen_BA.state_dict(),
                           args.results_gen_model_dir + '/%d-%d_gen_BA.pkl' % (epoch, batches_done))
                torch.save(self.dis_A.state_dict(),
                           args.results_dis_model_dir + '/%d-%d_dis_A.pkl' % (epoch, batches_done))
                torch.save(self.dis_B.state_dict(),
                           args.results_dis_model_dir + '/%d-%d_dis_B.pkl' % (epoch, batches_done))
                save_image(fake_gt, '%s/%s-%s.bmp' % (args.results_gt_dir, epoch, batches_done), nrow=4, normalize=True)
                save_image(fake_img, '%s/%s-%s.bmp' % (args.results_img_dir, epoch, batches_done), nrow=4,
                           normalize=True)

        f.close()
        writer.close()




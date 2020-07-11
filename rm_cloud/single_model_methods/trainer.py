#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""
import os
import torch
import time
from torch.autograd import Variable
from lib.loss.mse_loss import get_mse_loss_function
from lib.utils.convert_RGB import convert
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from lib.utils.log_report import LogReport


class Trainer(object):
    """
        Each trainer should  inherit this base trainer

    """
    def __init__(self,model,optimizer,):
        self.model = model.train()
        self.optimizer = optimizer

    def train_single(self,args,data_loader,start_epoch,end_epoch,):

        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.results_model_dir, exist_ok=True)
        os.makedirs(args.results_img_dir, exist_ok=True)

        begin_time = time.time()
        print("======== begin train model ========")
        print('data_size:',len(data_loader))
        writer = SummaryWriter(log_dir='{}'.format(args.log_dir),comment='train_loss')

        best_loss = 1
        for epoch in range(start_epoch,end_epoch):
            for i, (img,gt,name) in enumerate(data_loader):
                if torch.cuda.is_available():
                    img = Variable(img.cuda())
                    gt = Variable(gt.cuda())
                else:
                    img = Variable(img)
                    gt = Variable(gt)

                gen_img, loss = self.optimize_strategy(img,gt)

                gen_RGB = convert(gen_img[0])

                batches_done = epoch*len(data_loader) + i

                # save best model and img
                if loss.data.cpu() < best_loss:
                    best_loss = loss.data.cpu()
                    torch.save(self.model.state_dict(), args.results_model_dir +'/%d-%d_best_model.pkl'%(epoch,batches_done))
                    gen_RGB.save(args.results_img_dir +'%d-%d.png' %(epoch,batches_done))

                writer.add_scalar('{}_train_loss'.format(args.model.arch),loss.cpu(), batches_done)

                # print log
                if i % args.interval == 0:
                    print('[epoch %d/%d] [batch %d/%d] [loss: %f]' %(epoch,end_epoch,batches_done,(end_epoch*len(data_loader)),loss.item()))

                # write log in txt
                f = open(os.path.join(args.log_dir, 'log.txt'), 'a+')
                info = 'epoch:' + str(epoch) + ' ' + 'batches_done:' + str(batches_done) +  ' ' + 'loss:' + str(loss.data.cpu())
                f.write(info + '\n')
            if epoch > end_epoch-50:
                torch.save(self.model.state_dict(), args.results_model_dir +'/%d-%d.pkl'%(epoch,batches_done))
                gen_RGB.save(args.results_img_dir + '%d-%d.png' % (epoch, batches_done))

        print("total_consum_time = %.2f s" % (time.time()-begin_time))
        writer.close()
        f.close()


    def train_multi(self,args,data_loader,start_epoch,end_epoch,):

        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.results_model_dir, exist_ok=True)
        os.makedirs(args.results_img_dir, exist_ok=True)
        begin_time = time.time()

        print("=======training model========")
        print("======== begin train model ========")
        print('data_size:',len(data_loader))
        writer = SummaryWriter(log_dir='{}'.format(args.log_dir),comment='train_loss')
        best_loss = args.best_loss

        for epoch in range(start_epoch,end_epoch):
            for i, (img_0,gt_0,img_1,gt_1,img_2,gt_2,name) in enumerate(data_loader):
                if torch.cuda.is_available():
                    img_0 = Variable(img_0.cuda())
                    gt_0 = Variable(gt_0.cuda())
                    img_1 = Variable(img_1.cuda())
                    gt_1 = Variable(gt_1.cuda())
                    img_2 = Variable(img_2.cuda())
                    gt_2 = Variable(gt_2.cuda())
                else:
                    img_0 = Variable(img_0)
                    gt_0 = Variable(gt_0)
                    img_1 = Variable(img_1)
                    gt_1 = Variable(gt_1)
                    img_2 = Variable(img_2)
                    gt_2 = Variable(gt_2)

                gen_img,loss = self.optimize_multi_strategy(img_0,gt_0,img_1,gt_1,img_2,gt_2)
                gen_RGB = convert(gen_img[2][0])#[2]->256*256  [0]->batch

                batches_done = epoch*len(data_loader) + i

                if loss.data.cpu() < best_loss:
                    best_loss = loss.data.cpu()
                    torch.save(self.model.state_dict(),
                    args.results_model_dir + '/%d-%d_best_model.pkl' % (epoch, batches_done))
                    gen_RGB.save(args.results_img_dir + '%d-%d_best.png' % (epoch, batches_done))

                writer.add_scalar('{}_train_loss'.format(args.model.arch),loss.cpu(), batches_done)


                if i % args.interval == 0:
                    print('[epoch %d/%d] [batch %d/%d] [loss: %f]' %(epoch,end_epoch,batches_done,(end_epoch*len(data_loader)),loss.item()))

                f = open(os.path.join(args.log_dir, 'log.txt'), 'a+')
                info = 'epoch:' + str(epoch) + ' ' + 'batches_done:' + str(batches_done) + ' ' + 'loss:' + str(loss.cpu())
                f.write(info + '\n')
            if epoch > end_epoch - 50:
                torch.save(self.model.state_dict(), args.results_model_dir + '/%d-%d.pkl' % (epoch, batches_done))
                gen_RGB.save(args.results_img_dir + '%d-%d.png' % (epoch, batches_done))
            # if epoch % 5000== 0:
            #     torch.save(self.model.state_dict(),args.results_model_dir + '/%d-%d.pkl' % (epoch, batches_done))
            #     gen_RGB.save(args.results_img_dir + '%d-%d.png' % (epoch, batches_done))

        print("total_consum_time = %.2f s" % (time.time()-begin_time))

        f.close()
        writer.close()



    def optimize_strategy(self,img,gt):
        # train generator#
        self.optimizer_G.zero_grad()
        gen_imgs = self.model(img)
        mse_loss = get_mse_loss_function()
        loss = mse_loss(gen_imgs,gt)

        #BP
        loss.backward()
        self.optimizer_G.step()

        return gen_imgs,loss

    def optimize_multi_strategy(self,img_0,gt_0,img_1,gt_1,img_2,gt_2):

        gen_imgs = self.model(img_0,img_1,img_2)
        mse_loss = get_mse_loss_function()
        loss_0 = mse_loss(gen_imgs[2],gt_0)  #256
        loss_1 = mse_loss(gen_imgs[1],gt_1)  #128
        loss_2 = mse_loss(gen_imgs[0],gt_2)  #64
        loss = loss_0 + loss_1 + loss_2

        #BP
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return gen_imgs,loss







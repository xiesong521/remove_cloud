#!/user/bin/python3
#coding:utf-8
"""
    Author:XieSong
    Email:18406508513@163.com
    
    Copyright:XieSong
    Licence:MIT

"""

#coding:utf-8
"""
划分数据集：
train:test=8：2
"""

import glob
import os
import random
import shutil

Dataset_dir = "E:/landsat_dataset/datasat_clip/"
train_cloud_dir = "E:/landsat_dataset/landsat8/real_train_cloud/"
train_free_dir = "E:/landsat_dataset/landsat8/real_train_free/"

test_cloud_dir = "E:/landsat_dataset/landsat8/real_test_cloud/"
test_free_dir = "E:/landsat_dataset/landsat8/real_test_free/"

#设置比例
train_per = 0.8
test_per = 0.2

listdir = os.listdir(Dataset_dir)#1-10文件夹
print(listdir)
lenlist = len(listdir) #因为有个文件是readme.txt
for i in range(lenlist):
    subpath = os.path.join(Dataset_dir, listdir[i]) #得到这些文件的相对路径
    listsubdir = os.listdir(subpath)#得到每个文件下的有云 无云文件
    lensubdir = len(listsubdir)
    for j in range(lensubdir):
        datadir = os.path.join(subpath, listsubdir[j])
        for root, dirs, files in os.walk(datadir):  # 遍历每一个波段
            for dir in dirs:
                # dirname = root.split('/')[-1].split('\\')[-1]  # 根据有云无云去分
                # name = root.split('/')[4].split("\\")[0]  # 获取第几个文件名 1,3 做测试
                img_list = glob.glob(os.path.join(root, dir, '*.bmp'))#glob.glob获取指定目录下的所有文件
                len_image = len(img_list)#该文件下的文件数量
                random.seed(0)  # 设置随机种子使得每一次产生的随机数相同
                random.shuffle(img_list)  # 打乱图片的顺序
                img_num = len(img_list)
                train_point = int(img_num * train_per)
                trainA_outpath = train_cloud_dir + dir + '/'
                testA_outpath = test_cloud_dir + dir + '/'
                trainB_outpath = train_free_dir + dir + '/'
                testB_outpath = test_free_dir + dir + '/'
                os.makedirs(trainA_outpath,exist_ok=True)
                os.makedirs(testA_outpath,exist_ok=True)
                os.makedirs(trainB_outpath,exist_ok=True)
                os.makedirs(testB_outpath,exist_ok=True)
                for m in range(img_num):
                    if j == 0:#有云
                        if m < train_point:
                            out_dir = train_cloud_dir + dir + '/'

                        else:
                            out_dir = test_cloud_dir + dir + '/'
                        out_path = out_dir + listdir[i] + '_'+ os.path.split(img_list[m])[-1]
                        shutil.copy(img_list[m],out_path)  #将每一张图片复制到新的位置
                    if j == 1:  # 无云
                        if m < train_point:
                            out_dir = train_free_dir + dir + '/'

                        else:
                            out_dir = test_free_dir + dir + '/'
                        out_path = out_dir + listdir[i] + '_'+ os.path.split(img_list[m])[-1]
                        shutil.copy(img_list[m], out_path)  # 将每一张图片复制到新的位置









#!/user/bin/python3
#coding:utf-8
"""
    Author:XieSong
    Email:18406508513@163.com
    
    Copyright:XieSong
    Licence:MIT

"""
import os
root = 'E:\landsat_dataset\datasat_clip\\199025\\'
dirs = os.listdir(root)
for dir in dirs:
    bandlist = os.listdir(os.path.join(root,dir))
    for band in bandlist:
        print(os.path.join(root,dir,band))
        print(len(os.listdir(os.path.join(root,dir,band))))

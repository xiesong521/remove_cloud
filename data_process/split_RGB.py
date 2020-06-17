#!/user/bin/python3
#coding:utf-8
"""
    Author:XieSong
    Email:18406508513@163.com
    
    Copyright:XieSong
    Licence:MIT

"""
import os
from PIL import Image
def split_channels(rgb_dir,dest_dir):

    for file in os.listdir(rgb_dir):
        img = Image.open(os.path.join(rgb_dir,file))
        r,g,b = img.split()
        r.save(os.path.join(dest_dir,'B4',file))
        g.save(os.path.join(dest_dir,'B3',file))
        b.save(os.path.join(dest_dir,'B2',file))






if __name__== '__main__':
    rgb_dir = 'C:\\Users\\xs\Desktop\师兄数据集_跑的结果\传统_408\\HF\RGB'
    dest_dir = 'C:\\Users\\xs\Desktop\师兄数据集_跑的结果\传统_408\\HF\\'

    split_channels(rgb_dir, dest_dir)
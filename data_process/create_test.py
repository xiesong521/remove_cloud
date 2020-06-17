#!/user/bin/python3
#coding:utf-8
"""
    Author:XieSong
    Email:18406508513@163.com
    
    Copyright:XieSong
    Licence:MIT

"""
import os
import shutil
def generate_path(dest_dir):
    os.makedirs(os.path.join(dest_dir, 'RGB'), exist_ok=True)
    for i in range(1,12):
        if i != 8:
            os.makedirs(os.path.join(dest_dir,'B{}'.format(i)),exist_ok=True)

def find_special_file(name_list,src_dir,dest_dir):
    list_dir = os.listdir(src_dir)
    for dir in list_dir:
        file_list = os.listdir(src_dir+dir)
        for name in name_list:
            if name in file_list:
                shutil.copy(os.path.join(src_dir,dir,name),os.path.join(dest_dir,dir,name))


if __name__ == '__main__':
    dest_dir = 'D:\\RSC\\'
    generate_path(dest_dir)

    src_dir = 'E:\师兄数据集_跑的结果\师兄论文跑的结果\\'
    name_list = ['001_566.bmp','003_748.bmp']
    find_special_file(name_list,src_dir,dest_dir)
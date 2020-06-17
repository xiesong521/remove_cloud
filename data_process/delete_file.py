#!/user/bin/python3
#coding:utf-8
"""
    Author:XieSong
    Email:18406508513@163.com
    
    Copyright:XieSong
    Licence:MIT

"""
import os
def delete_img(root , name_list):
    bands = os.listdir(root)
    for band in bands:
        for name in name_list:
            print("remove {} from {}".format(name,band))
            name = name + '.bmp'
            os.remove(os.path.join(root,band,name))



if __name__ == '__main__':
    root1 = 'E:/study_resource/dataset/125033/cloud/'
    root2 = 'E:/study_resource/dataset/125033/free/'
    name_list = ['61','62','63','64','65','90','91','92','93']

    delete_img(root1,name_list)
    delete_img(root2,name_list)
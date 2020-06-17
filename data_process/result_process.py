#!/user/bin/python3
#coding:utf-8
"""
    Author:XieSong
    Email:18406508513@163.com
    
    Copyright:XieSong
    Licence:MIT

"""
import os
root = 'E:\study_resource\\rm_cloud\experiments\ssa\指标\\'
list_xls = os.listdir(root)

import pandas as pd

# excelFile = 'E:\study_resource\\rm_cloud\experiments\ms\zhibiao\\1ms.xls'
psnr = 'E:\study_resource\\rm_cloud\experiments\ssa\指标\\psnr.txt'
ssim = 'E:\study_resource\\rm_cloud\experiments\ssa\指标\\ssim.txt'
# 2,861,1,860
# 将每一个表格中的数据求平均，31个表格里的放一起
for xl in list_xls:
    df = pd.read_excel(os.path.join(root,xl))
    sum = 0
    for i in range(1, 860, 3):
        x = df.loc[i]
        sum += x
    sum = sum / 287
    with open(psnr, 'a') as f:
        f.write(str(sum))
    print(sum)
    sum = 0
f.close()









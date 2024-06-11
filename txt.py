'''
Author: yehongcen && as_yhc@163.com
Date: 2024-04-23 15:36:38
LastEditors: redust as_yhc@163.com
LastEditTime: 2024-04-23 16:26:41
FilePath: \lastdance\txt.py
Description: 生成训练集txt
Copyright (c) 2024 by yehongcen, All Rights Reserved. 
'''
import os

path1 = 'D:\\speechdata\\vctk_voice\\clean_testset_wav\\' # 带噪语音路径
path2 = 'D:\\speechdata\\vctk_voice\\clean_testset_wav\\' # 干净语音路径

path_write = 'D:\\yhc\BWE\\vctktestset.txt' # txt文件写入路径(包含txt文件名)

files1 = os.listdir(path1)
files1.sort()
files2 = os.listdir(path2)
files2.sort()
print(files1)
print(files2)

with open(path_write, 'w+') as txt:
    for i in range(len(files1)):
    # for i in range(3000):
        string = path1 + files1[i] + ' ' + path2 + files2[i]
        print('string', string)
        txt.write(string + '\n')

txt.close()
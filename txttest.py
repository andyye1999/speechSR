'''
Author: yehongcen && as_yhc@163.com
Date: 2024-04-23 15:36:38
LastEditors: redust as_yhc@163.com
LastEditTime: 2024-04-23 16:27:20
FilePath: \lastdance\txttest.py
Description: 生成测试集txt
Copyright (c) 2024 by yehongcen, All Rights Reserved. 
'''
import os

path1 = 'D:\\speechdata\\vctk_voice\\clean_testset_wav\\'  # need to enhance dir


path_write = 'D:\\yhc\BWE\\vctkenhanceset.txt' # txt文件写入路径(包含txt文件名)

files1 = os.listdir(path1)

print(files1)


with open(path_write, 'w+') as txt:
    for i in range(len(files1)):
        string = path1 + files1[i]
        print('string', string)
        txt.write(string + '\n')

txt.close()
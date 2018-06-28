# -*- coding: utf-8 -*-

import numpy as np

def readFile(path):
    # 打开文件（注意路径）
    f = open(path)
    # 逐行进行处理
    first_ele = True
    for data in f.readlines():
        ## 去掉每行的换行符，"\n"
        data = data.strip('\n')
        ## 按照 空格进行分割。
        nums = data.split("\t")
        ## 添加到 matrix 中。
        if first_ele:
            ### 将字符串转化为整型数据
            nums = [x for x in nums ]
            ### 加入到 matrix 中 。
            matrix = np.array(nums)
            first_ele = False
        else:
            nums = [x for x in nums]
            matrix = np.c_[matrix,nums]
    dealMatrix(matrix)
    f.close()

def dealMatrix(matrix):
    ## 一些基本的处理。
    print("transpose the matrix")
    matrix = matrix.transpose()
    print(matrix)
    #print("matrix trace ")
    #print(np.trace(matrix))

readFile('./fold_0_data.txt')

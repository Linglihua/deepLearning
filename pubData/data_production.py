# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：data_production.py
@Author  ：Xin Zheng
@Date    ：2021/7/31 16:55 
'''
import numpy as np

def make_data(nDim):
    x0 = np.linspace(1, np.pi, 50)                                   #一个维度的特征
    x = np.vstack([[x0, ], [i**x0 for i in range(2, nDim+1)]])         #nDim个维度的特征
    y = np.sin(x0) + np.random.normal(0, 0.15, len(x0))
    return x.transpose(), y
# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：AffinityPropagationC.py
@Author  ：Xin Zheng
@Date    ：2021/8/14 15:18 
'''

import numpy as np
from sklearn.cluster import AffinityPropagation

class AffinityPropagationC:
    __X = []
    def __init__(self):
        self.__X = np.array([[1, 2], [1, 4],[0.7, 0], [0.2, 5], [0, 4], [1.3, 0], [0.1, 2], [0, 4], [0.4, 0]])

    def affinityPropagation(self):
        af = AffinityPropagation(preference=-5, ).fit(self.__X) #
        print("label: ", af.labels_) #查看聚类结果
        af2 = AffinityPropagation(preference=-8, ).fit(self.__X)
        print("label: ", af2.labels_)  # 查看聚类结果
        print("n_iter: ", af2.n_iter_) #得迭代次数
        print("cluster_center: ", af2.cluster_centers_)#质心坐标
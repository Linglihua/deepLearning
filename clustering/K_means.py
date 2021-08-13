# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：K_means.py
@Author  ：Xin Zheng
@Date    ：2021/8/13 15:58 
'''

import numpy as np
from sklearn.cluster import KMeans



class K_means:
    __X = []

    def __init__(self):
        self.__X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [1, 0], [4, 2], [4, 4], [4, 0]])

    def k_means(self):
        kmeans = KMeans(n_clusters=2, random_state=0).fit(self.__X)
        print("kmeans.labels: ", kmeans.labels_)       #训练数据的聚类结果
        print("pridict: ", kmeans.predict([[0, 0], [4, 4]]))   #预测新数据的类别
        print("center: ", kmeans.cluster_centers_) #中心点坐标
        print("inertia: ", kmeans.inertia_)#收敛值
        print("transform: ", kmeans.transform([[4, 2], [4, 4]])) #查询两个向量与每个中心点的距离
        print("score: ", kmeans.score([[2, 2], [5, 3]]))  #测试数据与中心点的距离
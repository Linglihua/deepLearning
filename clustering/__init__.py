# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：__init__.py.py
@Author  ：Xin Zheng
@Date    ：2021/8/13 15:56 
'''

import K_means
import AffinityPropagationC


if __name__ == '__main__':
    #K-means
    # k_means = K_means.K_means()
    # k_means.k_means()

    #邻近算法
    ap = AffinityPropagationC.AffinityPropagationC()
    ap.affinityPropagation()
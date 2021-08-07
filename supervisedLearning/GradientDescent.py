# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：GradientDescent.py
@Author  ：Xin Zheng
@Date    ：2021/8/6 13:52 
'''
import random

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
import numpy as np


class GradientDescent:

    __X = [[0, 0], [2, 1], [5, 4]]  #样本特征
    __Y = [0, 2, 2]

    def SGDRegressor(self):
        reg = SGDRegressor(penalty="elasticnet", max_iter=1000)
        reg.fit(self.__X, self.__Y)

        print("predict: ", reg.predict([[4, 3]]))
        print("coef_: ", reg.coef_)
        print("intercept_: ", reg.intercept_)\

    '''
    期预测的结果是训练目标中的值
    '''
    def SGDClassifier(self):
        clf = SGDClassifier(penalty="elasticnet", max_iter=100)
        clf.fit(self.__X, self.__Y)

        print("predict: ", clf.predict([[4, 3]]))
        print("coef_: ", clf.coef_)
        print("intercept_: ", clf.intercept_)

    '''
    partial_fit在之前训练的结果上继续进行
    '''
    def incrementalLearning(self):
        reg = SGDRegressor(loss="squared_loss", penalty="none", tol=1e-5)
        X = np.linspace(0, 1, 50)
        Y = X/2 + 0.3 + np.random.normal(0, 0.15, len(X))
        X = X.reshape(-1, 1)

        for i in range(10000):
            idx = random.randint(0, len(Y) - 1)
            reg.partial_fit(X[idx: idx+10], Y[idx: idx+10])

        print("coef_: ", reg.coef_)
        print("intercept_: ", reg.intercept_)
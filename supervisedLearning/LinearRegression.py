# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：LinearRegression.py
@Author  ：Xin Zheng
@Date    ：2021/8/2 22:49 
'''

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


class linearRegression:
    dims = [1, 3, 6, 12]
    x, y = [], []
    def __init__(self):
        self.x, self.y = self.__make_data(self.dims[-1])


    #最小二乘法
    def func_ols(self):
        self.x = np.array([[0, 1], [3, -2], [2, 3]])  # 训练样本的特征
        self.y = np.array([0.5, 0.3, 0.9])

        reg = linear_model.LinearRegression()  # 初始化对象
        reg.fit(self.x, self.y)  # 开始训练

        print("intercept_: ", reg.intercept_)  # 截距
        print("coef_: ", reg.coef_)  # 参数

        reg.predict([[1, 2], [-3, 2]])  # 预测

    '''
    因为OLS始终在试图最小化方差，因此为了更好的拟合训练数据中很小的x值差异产生的较大的y值差异，
    必须使用较大的w值。而越来越大的w值在测试数据上反映出的实验结果则是任何一个特征微小的变化都
    会导致最终预测目标值的大幅度变化，即过度拟合
    '''

    def ols_lack(self):

        for idx, i in enumerate(self.dims):
            plt.subplot(2, len(self.dims) / 2, idx + 1)
            reg = linear_model.LinearRegression()

            sub_x = self.x[:, 0:i]  # 取m至n-1列数据
            reg.fit(sub_x, self.y)
            plt.plot(self.x[:, 0], reg.predict(sub_x))
            plt.plot(self.x[:, 0], self.y, ".")
            plt.title("dim=%s" % i)

            print("dim %d: " % i)
            print("intercept_: %s" % (reg.intercept_,))
            print("coef_: %s" % (reg.coef_,))
        plt.show()

    '''
    通过改变回归目标函数，达到回归控制参数随着维度疯狂增长的目的
    '''
    def ridge_regression(self):
        alphas = [1e-15, 1e-12, 1e-5, 1, ]  # a参数
        for idx, i in enumerate(alphas):
            plt.subplot(2, len(alphas) / 2, idx + 1)
            reg = linear_model.Ridge(alpha=i)  # 岭回归模型

            sub_x = self.x[:, 0: 12]  # 取出全部12维数据
            reg.fit(sub_x, self.y)
            plt.plot(self.x[:, 0], reg.predict(sub_x))
            plt.plot(self.x[:, 0], self.y, ".")
            plt.title("dim=12, alpha=%e" % i)

            print("alpha %e :" % i)
            print("intercept_: %s" % (reg.intercept_,))
            print("coef_: %s" % (reg.coef_,))
        plt.show()

    def lasso_regression(self):
        alphas = [1e-10, 1e-3, 1, 10, ]
        for idx, i in enumerate(alphas):
            plt.subplot(2, len(alphas) / 2, idx + 1)
            reg = linear_model.Lasso(alpha=i)
            sub_x = self.x[:, 0:12]
            reg.fit(sub_x, self.y)
            plt.plot(self.x[:, 0], reg.predict(sub_x))
            plt.plot(self.x[:, 0], self.y, ".")
            plt.title("dim=12, alpha=%e" % i)

            print("alpha %e: " % i)
            print("intercept_: %s" % (reg.intercept_,))
            print("coef_: %s" % (reg.coef_,))
        plt.show()

    def __make_data(self, nDim):
        x0 = np.linspace(1, np.pi, 50)  # 一个维度的特征
        self.x = np.vstack([[x0, ], [i ** x0 for i in range(2, nDim + 1)]])  # nDim个维度的特征
        self.y = np.sin(x0) + np.random.normal(0, 0.15, len(x0))
        return self.x.transpose(), self.y
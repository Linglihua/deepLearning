# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：GaussianProcessRegressor.py
@Author  ：Xin Zheng
@Date    ：2021/8/9 16:07 
'''

import numpy as np
from matplotlib import pyplot as plt
from sklearn .gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C, Product



class GuassianProcessRegressor:
    __X = None
    __y = None
    __x = None
    def __init__(self):
        self.__X = np.linspace(0, 10, 20)  # 20个训练样本的特征值
        self.__y = self.__f() + np.random.normal(0, 0.5, self.__X.shape[0]) #样本目标值，并加入噪声
        self.__x = np.linspace(0, 10, 20) #测试样本特征值

    def gaussianProcessReg(self):
        #定义两个核函数并取它们的积
        kernel = Product(C(0.1), RBF(10, (1e-2, 1e2)))
        #初始化模型： 传入核函数对象，优化次数。噪声超参数
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0.3)
        gp.fit(self.__X.reshape((-1, 1)), self.__y)
        y_pred, sigma = gp.predict(self.__x.reshape(-1, 1), return_std=True)

        fig = plt.figure()
        plt.plot(self.__x, self.__f(), 'r:', label=u'$f(x) = x\,\sin(x) - x$')
        plt.plot(self.__x, self.__y, 'r.', markersize=10, label=u'Observation')
        plt.plot(self.__x, y_pred, 'b-', label=u'Prediction')
        plt.fill(np.concatenate([self.__x, self.__x[:: -1]]),
                 np.concatenate([y_pred - 2 * sigma, (y_pred + 2 * sigma) [:: -1]]),
                 alpha=.3, fc='b', label='95% confidence'
                 )
        plt.legend(loc='lower left')
        plt.show()



    def __f(self):
        return self.__X*np.sin(self.__X) - self.__X


# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：ridge_reg.py
@Author  ：Xin Zheng
@Date    ：2021/7/31 16:39 
'''
import matplotlib.pyplot as plt
from sklearn import linear_model

from pubData.data_production import make_data


def ridge_regression(nDims):
    alphas = [1e-15, 1e-12, 1e-5, 1, ] #a参数
    x, y = make_data(nDims[-1])
    for idx, i in enumerate(alphas):
        plt.subplot(2, len(alphas)/2, idx + 1)
        reg = linear_model.Ridge(alpha = i)  #岭回归模型

        sub_x = x[:, 0 : 12] #取出全部12维数据
        reg.fit(sub_x, y)
        plt.plot(x[:,0], reg.predict(sub_x))
        plt.plot(x[:,0], y, ".")
        plt.title("dim=12, alpha=%e" %i)

        print("alpha %e :" %i)
        print("intercept_: %s" % (reg.intercept_))
        print("coef_: %s" % (reg.coef_))
    plt.show()



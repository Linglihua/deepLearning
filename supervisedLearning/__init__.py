# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：__init__.py
@Author  ：Xin Zheng
@Date    ：2021/8/3 9:55 
'''
import LinearRegression
import GradientDescent
if __name__ == '__main__':
    #最小二乘法
    # linearR = LinearRegression.linearRegression()
    # # linearR.func_ols()
    # # linearR.ols_lack()
    # #linearR.ridge_regression()
    # linearR.lasso_regression()

    #梯度回归
    gradientDesc = GradientDescent.GradientDescent()
    gradientDesc.SGDRegressor()
    gradientDesc.SGDClassifier()
    gradientDesc.incrementalLearning()


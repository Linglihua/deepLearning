# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：__init__.py
@Author  ：Xin Zheng
@Date    ：2021/8/3 9:55 
'''
import LinearRegression
import GradientDescent
import SVM
import NaiveBayes

if __name__ == '__main__':
    #最小二乘法
    #linearR = LinearRegression.linearRegression()
    # # linearR.func_ols()
    #linearR.ols_lack()
    # #linearR.ridge_regression()
    # linearR.lasso_regression()

    #梯度回归
    # gradientDesc = GradientDescent.GradientDescent()
    # gradientDesc.SGDRegressor()
    # gradientDesc.SGDClassifier()
    # gradientDesc.incrementalLearning()

    #SVM支持向量机
    # svmer = SVM.SVMer()
    # svmer.svmFun()

    #朴素贝叶斯
    naiveBayes = NaiveBayes.NaiveBayes()
    #naiveBayes.guassianNaiveBayes()
    naiveBayes.bernolliNaiveBayes()



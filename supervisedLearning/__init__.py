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
import GaussianProcessRegressor
import DecisionTreeClassifier
import EnsembleLearning
import GridSearchCV

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
    # naiveBayes = NaiveBayes.NaiveBayes()
    #naiveBayes.guassianNaiveBayes()
    # naiveBayes.bernolliNaiveBayes()

    #高斯过程
    # gpr = GaussianProcessRegressor.GuassianProcessRegressor()
    # gpr.gaussianProcessReg()

    #决策树
    # dtc = DecisionTreeClassifier.DecisionTreeClassifier()
    # dtc.decisionTreeCla()

    #集成学习
    # ensembleL = EnsembleLearning.EnsembleLearning()
    # ensembleL.randomForestClassifier()
    # ensembleL.adaBoostClassifier()

    #GridSearchCV
    gridSearchCV = GridSearchCV.GridSearchCVer()
    gridSearchCV.gridSearchCV()
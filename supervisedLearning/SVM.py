# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：SVM.py
@Author  ：Xin Zheng
@Date    ：2021/8/7 15:29 
'''

from sklearn import svm

class SVMer:
    __X = [[0, 0], [2, 2]]
    __Y = [1, 2]
    def svmFun(self):
        clf = svm.SVC(kernel="rbf") #初始化使用径向基核的分类器
        clf.fit(self.__X, self.__Y)

        t = [[2, 1], [0, 1]]
        print("predict: ", clf.predict(t))
        print("decision: ",clf.decision_function(t))
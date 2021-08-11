# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：EnsembleLearning.py
@Author  ：Xin Zheng
@Date    ：2021/8/11 14:13 
'''
import random
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

class EnsembleLearning:
    __iris = load_iris()

    def randomForestClassifier(self):
        # 基模型数量，使用有放回采样，训练后进行OOB测试
        clf = RandomForestClassifier(n_estimators=20, bootstrap=True, oob_score=True)
        clf.fit(self.__iris.data, self.__iris.target)
        print("score: ", clf.oob_score_)

    def adaBoostClassifier(self):
        random.shuffle(self.__iris.data)
        random.shuffle(self.__iris.target)
        clf = AdaBoostClassifier(GaussianNB())
        clf.fit(self.__iris.data[: -20], self.__iris.target[: -20])
        print("score: ", clf.score(self.__iris.data[-20: ], self.__iris.target[-20: ]))

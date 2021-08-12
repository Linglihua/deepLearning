# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：GridSearchCV.py
@Author  ：Xin Zheng
@Date    ：2021/8/12 16:23 
'''

from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

class GridSearchCVer:
    __iris = datasets.load_iris()

    def gridSearchCV(self):
        #待调试超参数列表
        tuned_parameters =[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                            'C': [1, 10, 100, 1000]},
                           {'kernel': ['linear'], 'C': [1,10,100,1000]}
                           ]
        #初始化GSCV，传入待调试，模型，超参数列表，验证N-Fold值，验证标准
        clf = GridSearchCV(estimator=SVC(), param_grid=tuned_parameters, cv=10, scoring='accuracy')
        clf.fit(self.__iris.data,self.__iris.target)

        print("Best parameter set found om development set: ")
        print()
        print("最佳超参数: ", clf.best_params_)
        print()
        print("Grid scores on development set: ")
        print()

        means = clf.cv_results_['mean_test_score'] #评价分值
        stds = clf.cv_results_['std_test_score'] #分值标准差
        durations = clf.cv_results_['mean_fit_time'] #训练时间

        for mean, std, duration, params in zip(means, stds, durations, clf.cv_results_['params']):
            print("%0.3f (+/%0.03f) for %r in %f seconds" % (mean, std * 2, params, duration))
            print()
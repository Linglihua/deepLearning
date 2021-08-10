# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：DecisionTreeClassifier.py
@Author  ：Xin Zheng
@Date    ：2021/8/10 14:57 
'''
from sklearn import tree
import graphviz

class DecisionTreeClassifier:
    __X = []
    __Y = []

    def __init__(self):
        self.__X = [[20, 30000, 400],
                    [37, 13000, 0],
                    [50, 26000, 0],
                    [28, 10000, 3000],
                    [31, 19000, 1500000],
                    [46, 7000, 6000]
                    ]
        self.__Y = [1, 0, 0, 0, 1, 0]

    def decisionTreeCla(self):
        clf = tree.DecisionTreeClassifier(criterion="entropy")
        clf = clf.fit(self.__X, self.__Y)
        print("predict: ", clf.predict([[40, 6000, 0]]))
        print("feature_importances_: ", clf.feature_importances_)  #查看特征的重要性

        self.__treeShow(clf)

    def __treeShow(self, clf):
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=[u"年龄", u"收入", u"存款"],
                                        class_names=[u"普通", u"VIP"],
                                        filled=True, rotate=True
                                        )
        graph = graphviz.Source(dot_data)
        graph.render("mytree")  #保存为文件
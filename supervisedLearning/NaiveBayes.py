# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：NaiveBayes.py
@Author  ：Xin Zheng
@Date    ：2021/8/8 14:57 
'''

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

class NaiveBayes:

    def guassianNaiveBayes(self):
        '''
        。这个数据集里一共包括150行记录，其中前四列为花萼长度，花萼宽度，花瓣长度，花瓣宽度等4个用于
        识别鸢尾花的属性，第5列为鸢尾花的类别（包括Setosa，Versicolour，Virginica三类）。
        也即通过判定花萼长度，花萼宽度，花瓣长度，花瓣宽度的尺寸大小来识别鸢尾花的类别。
        '''
        iris = datasets.load_iris()  #导入鸢尾花数据集
        print("iris.data.shape: ", iris.data.shape)  # (150, 4)
        print("iris.feature_names: ", iris.feature_names)  # [花萼长，花萼宽，花瓣长，花瓣宽]

        gnb = GaussianNB()#引入高斯朴素贝叶斯模型，初始化模型对象
        gnb.fit(iris.data, iris.target)#训练
        print("class_prior: ", gnb.class_prior_)#查看先验概率（标签及其概率）
        print("calss_count: ", gnb.class_count_)#训练集标签数量以及每种标签的数量
        #由于数据有四维特征，并且有三种标签，所以训练产生12个高斯模型
        print("theta_: ", gnb.theta_)
        print("sigma: ", gnb.sigma_)

    def bernolliNaiveBayes(self):
        clf = BernoulliNB(binarize=1) #阈值

        X = [[0.3, 0.2], [1.3, 1.2], [1.1, 1.2]]  #BernoulliNB内部会把用阈值1将其转化为二值
        Y = [0, 1, 1]
        clf.fit(X, Y)
        print("predict: ", clf.predict([[0.99, 0.99]]))
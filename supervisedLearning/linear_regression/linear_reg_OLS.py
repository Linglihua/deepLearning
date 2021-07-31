import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

from pubData.data_production import make_data


def func_ols():
    x = np.array([[0,1], [3,-2], [2,3]]) #训练样本的特征
    y = np.array([0.5, 0.3, 0.9])

    reg = linear_model.LinearRegression()  #初始化对象
    reg.fit(x, y) #开始训练

    print("intercept_: ", reg.intercept_)#截距
    print("coef_: ", reg.coef_)#参数

    reg.predict([[1,2], [-3,2]])  #预测

def ols_lack(nDims):
    x, y = make_data(nDims[-1])
    for idx, i in enumerate(nDims):

        plt.subplot(2, len(nDims)/2, idx+1)
        reg = linear_model.LinearRegression()

        sub_x = x[:, 0:i]   #取m至n-1列数据
        reg.fit(sub_x, y)
        plt.plot(x[:, 0], reg.predict(sub_x))
        plt.plot(x[:, 0], y, ".")
        plt.title("dim=%s"%i)

        print("dim %d: "%i)
        print("intercept_: %s"% (reg.intercept_, ))
        print("coef_: %s" %(reg.coef_, ))
    plt.show()

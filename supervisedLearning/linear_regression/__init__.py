# -*- coding: UTF-8 -*-
'''
@Project ：deepLearning 
@File    ：__init__.py.py
@Author  ：Xin Zheng
@Date    ：2021/7/31 16:03 
'''
import ridge_reg as ridge
import linear_reg_OLS as ols

if __name__ == '__main__':
    # 最小二乘法
    #ols.func_ols()
    #制造数据的维度
    dims = [1, 3, 6, 12]

    # 最小二乘法的缺陷展示
    #ols.ols_lack(dims)

    #岭回归函数
    ridge.ridge_regression(dims)


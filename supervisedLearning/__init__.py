import linear_reg_OLS as ols

if __name__ == '__main__':
    #ols.func_ols()  #最小二乘法

    dims = [1, 3, 6, 12]
    ols.ols_lack(dims) #最小二乘法的缺陷
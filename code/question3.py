import datetime
import numpy as np
import pandas as pd
import pylab as mpl
from matplotlib.font_manager import FontProperties
from pandas import DataFrame
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.metrics import mean_squared_error
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

fireRescue = pd.read_excel('2.xlsx', sheet_name='Sheet1', header=0)
sort = fireRescue['事件类别'].values
date = fireRescue['接警日期'].values
date = pd.DatetimeIndex(date)

def getSortTime(sortId):
    mon1 = 0
    mon2 = 0
    mon3 = 0
    mon4 = 0
    mon5 = 0
    mon6 = 0
    mon7 = 0
    mon8 = 0
    mon9 = 0
    mon10 = 0
    mon11 = 0
    mon12 = 0
    for i in range(len(date)):
        if sort[i]==sortId and date[i].month == 1:
            mon1 += 1
        if sort[i]==sortId and date[i].month == 2:
            mon2 += 1
        if sort[i]==sortId and date[i].month == 3:
            mon3 += 1
        if sort[i]==sortId and date[i].month == 4:
            mon4 += 1
        if sort[i]==sortId and date[i].month == 5:
            mon5 += 1
        if sort[i]==sortId and date[i].month == 6:
            mon6 += 1
        if sort[i]==sortId and date[i].month == 7:
            mon7 += 1
        if sort[i]==sortId and date[i].month == 8:
            mon8 += 1
        if sort[i]==sortId and date[i].month == 9:
            mon9 += 1
        if sort[i]==sortId and date[i].month == 10:
            mon10 += 1
        if sort[i]==sortId and date[i].month == 11:
            mon11 += 1
        if sort[i]==sortId and date[i].month == 12:
            mon12 += 1
    return [mon1//5,mon2//5,mon3//5,mon4//5,mon5//5,mon6//5,mon7//5,mon8//5,mon9//5,mon10//5,mon11//5,mon12//5]
sortId1=getSortTime("①")
sortId2=getSortTime("②")
sortId3=getSortTime("③")
sortId4=getSortTime("④")
sortId5=getSortTime("⑤")
sortId6=getSortTime("⑥")
sortId7=getSortTime("⑦")



mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
X=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
plt.plot(X, sortId3, color='yellow', marker='.', label='事件③')
plt.plot(X, sortId7, color='pink', marker='.', label='事件⑦')
plt.xlabel('月份')
plt.ylabel('平均各类事件发生次数')
X1=X.reshape([-1,1])


model = LinearRegression()
linear = linear_model.LinearRegression()           #建立回归方程拟合数据
linear.fit(X1, sortId3)
poly_reg = PolynomialFeatures(degree = 7)    #degree=2表示建立datas_X的二次多项式特征X_poly。然后创建线性回归，使用线性模型学习X_poly和datasets_Y之间的映射关系（即参数）
X_poly = poly_reg.fit_transform(X1)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, sortId3)
predict3=lin_reg_2.predict(poly_reg.fit_transform(X1))

predict3=np.array(predict3)

X2 = np.linspace(X.min(),X.max(),300) #300 represents number of points to make between T.min and T.max
predictSmooth = make_interp_spline(X, predict3)(X2)
plt.plot(X2, predictSmooth,color='yellow',label='事件③拟合')


linear = linear_model.LinearRegression()           #建立回归方程拟合数据
linear.fit(X1, sortId7)
poly_reg = PolynomialFeatures(degree = 5)    #degree=2表示建立datas_X的二次多项式特征X_poly。然后创建线性回归，使用线性模型学习X_poly和datasets_Y之间的映射关系（即参数）
X_poly = poly_reg.fit_transform(X1)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, sortId7)
predict7=lin_reg_2.predict(poly_reg.fit_transform(X1))

predict7=np.array(predict7)

X2 = np.linspace(X.min(),X.max(),300) #300 represents number of points to make between T.min and T.max
predictSmooth = make_interp_spline(X, predict7)(X2)
plt.plot(X2, predictSmooth,color="pink",label='事件⑦拟合')
plt.legend()
plt.show()


def get_lr_stats(yPredict, y, model):
    Residual = sum((y - yPredict) ** 2)  # 残差平方和
    total = sum((y - np.mean(y)) ** 2)  # 总体平方和
    R_square = 1 - Residual / total  # 相关性系数R^2
    message1 = ('相关系数(R^2)： ' + str(R_square))
    return print(message1)



plt.legend()
plt.show()

plt.plot(X, sortId1, color='green', marker='.', label='事件①')
plt.plot(X, sortId2, color='blue', marker='.', label='事件②')
plt.plot(X, sortId4, color='purple', marker='.', label='事件④')
plt.plot(X, sortId5, color='black', marker='.', label='事件⑤')
plt.plot(X, sortId6, color='orange', marker='.', label='事件⑥')



linear = linear_model.LinearRegression()           #建立回归方程拟合数据
linear.fit(X1, sortId1)
poly_reg = PolynomialFeatures(degree = 2)    #degree=2表示建立datas_X的二次多项式特征X_poly。然后创建线性回归，使用线性模型学习X_poly和datasets_Y之间的映射关系（即参数）
X_poly = poly_reg.fit_transform(X1)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, sortId1)
predict1=lin_reg_2.predict(poly_reg.fit_transform(X1))
predict1=np.array(predict1)
X2 = np.linspace(X.min(),X.max(),300) #300 represents number of points to make between T.min and T.max
predictSmooth = make_interp_spline(X, predict1)(X2)
plt.plot(X2, predictSmooth,color="green",label='事件①拟合')




linear = linear_model.LinearRegression()           #建立回归方程拟合数据
linear.fit(X1, sortId2)
poly_reg = PolynomialFeatures(degree = 8)    #degree=2表示建立datas_X的二次多项式特征X_poly。然后创建线性回归，使用线性模型学习X_poly和datasets_Y之间的映射关系（即参数）
X_poly = poly_reg.fit_transform(X1)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, sortId2)
predict2=lin_reg_2.predict(poly_reg.fit_transform(X1))
predict2=np.array(predict2)
X2 = np.linspace(X.min(),X.max(),300) #300 represents number of points to make between T.min and T.max
predictSmooth = make_interp_spline(X, predict2)(X2)
plt.plot(X2, predictSmooth,color="blue",label='事件②拟合')




linear = linear_model.LinearRegression()           #建立回归方程拟合数据
linear.fit(X1, sortId4)
poly_reg = PolynomialFeatures(degree = 6)    #degree=2表示建立datas_X的二次多项式特征X_poly。然后创建线性回归，使用线性模型学习X_poly和datasets_Y之间的映射关系（即参数）
X_poly = poly_reg.fit_transform(X1)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, sortId4)
predict4=lin_reg_2.predict(poly_reg.fit_transform(X1))
predict4=np.array(predict4)
X2 = np.linspace(X.min(),X.max(),300) #300 represents number of points to make between T.min and T.max
predictSmooth = make_interp_spline(X, predict4)(X2)
plt.plot(X2, predictSmooth,color="purple",label='事件④拟合')




linear = linear_model.LinearRegression()           #建立回归方程拟合数据
linear.fit(X1, sortId5)
poly_reg = PolynomialFeatures(degree = 5)    #degree=2表示建立datas_X的二次多项式特征X_poly。然后创建线性回归，使用线性模型学习X_poly和datasets_Y之间的映射关系（即参数）
X_poly = poly_reg.fit_transform(X1)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, sortId5)
predict5=lin_reg_2.predict(poly_reg.fit_transform(X1))
predict5=np.array(predict5)
X2 = np.linspace(X.min(),X.max(),300) #300 represents number of points to make between T.min and T.max
predictSmooth = make_interp_spline(X, predict5)(X2)
plt.plot(X2, predictSmooth,color="black",label='事件⑤拟合')



linear = linear_model.LinearRegression()           #建立回归方程拟合数据
linear.fit(X1, sortId6)
poly_reg = PolynomialFeatures(degree = 4)    #degree=2表示建立datas_X的二次多项式特征X_poly。然后创建线性回归，使用线性模型学习X_poly和datasets_Y之间的映射关系（即参数）
X_poly = poly_reg.fit_transform(X1)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, sortId6)
predict6=lin_reg_2.predict(poly_reg.fit_transform(X1))
predict6=np.array(predict6)
X2 = np.linspace(X.min(),X.max(),300) #300 represents number of points to make between T.min and T.max
predictSmooth = make_interp_spline(X, predict6)(X2)
plt.plot(X2, predictSmooth,color="orange",label='事件⑥拟合')


print("类型①拟合",end="")
get_lr_stats(predict1,sortId1 , model)
print("类型②拟合",end="")
get_lr_stats(predict2,sortId2 , model)
print("类型③拟合",end="")
get_lr_stats(predict3,sortId3 , model)
print("类型④拟合",end="")
get_lr_stats(predict4,sortId4 , model)
print("类型⑤拟合",end="")
get_lr_stats(predict5,sortId5 , model)
print("类型⑥拟合",end="")
get_lr_stats(predict6,sortId6 , model)
print("类型⑦拟合",end="")
get_lr_stats(predict7,sortId7 , model)



plt.legend()
plt.show()
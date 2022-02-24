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
populationArea = pd.read_excel('1.xlsx', sheet_name='Sheet1', header=0)

population=populationArea['人口（万人）'].values
area=populationArea['面积（km2）'].values
sort = fireRescue['事件类别'].values
region = fireRescue['事件所在的区域'].values




def getSortDensity(sortId,area):
    pieceA = 0
    pieceB = 0
    pieceC = 0
    pieceD = 0
    pieceE = 0
    pieceF = 0
    pieceG = 0
    pieceH = 0
    pieceI = 0
    pieceJ = 0
    pieceK = 0
    pieceL = 0
    pieceM = 0
    pieceN = 0
    pieceP = 0
    for i in range(len(sort)):
        if sort[i] == sortId and region[i] == 'A':
            pieceA += 1
        if sort[i] == sortId and region[i] == 'B':
            pieceB += 1
        if sort[i] == sortId and region[i] == 'C':
            pieceC += 1
        if sort[i] == sortId and region[i] == 'D':
            pieceD += 1
        if sort[i] == sortId and region[i] == 'E':
            pieceE += 1
        if sort[i] == sortId and region[i] == 'F':
            pieceF += 1
        if sort[i] == sortId and region[i] == 'G':
            pieceG += 1
        if sort[i] == sortId and region[i] == 'H':
            pieceH += 1
        if sort[i] == sortId and region[i] == 'I':
            pieceI += 1
        if sort[i] == sortId and region[i] == 'G':
            pieceJ += 1
        if sort[i] == sortId and region[i] == 'K':
            pieceK += 1
        if sort[i] == sortId and region[i] == 'L':
            pieceL += 1
        if sort[i] == sortId and region[i] == 'M':
            pieceM += 1
        if sort[i] == sortId and region[i] == 'N':
            pieceN += 1
        if sort[i] == sortId and region[i] == 'P':
            pieceP += 1
    return [pieceA/area[0],pieceB/area[1],pieceC/area[2],pieceD/area[3],pieceE/area[4],pieceF/area[5],pieceG/area[6],pieceH/area[7],pieceI/area[8],pieceJ/area[9],pieceK/area[10],pieceL/area[11],pieceM/area[12],pieceN/area[13]]
area=np.array(area)
N=6
SortDensity1=[round(j,N) for j in np.array(getSortDensity('①',area))/365/5*7]
SortDensity2=[round(j,N) for j in np.array(getSortDensity('②',area))/365/5*7]
SortDensity3=[round(j,N) for j in np.array(getSortDensity('③',area))/365/5*7]
SortDensity4=[round(j,N) for j in np.array(getSortDensity('④',area))/365/5*7]
SortDensity5=[round(j,N) for j in np.array(getSortDensity('⑤',area))/365/5*7]
SortDensity6=[round(j,N) for j in np.array(getSortDensity('⑥',area))/365/5*7]
SortDensity7=[round(j,N) for j in np.array(getSortDensity("⑦",area))/365/5*7]
sortD=[SortDensity1,SortDensity2,SortDensity3,SortDensity4,SortDensity5,SortDensity6,SortDensity7]
print(SortDensity1,SortDensity2,SortDensity3,SortDensity4,SortDensity5,SortDensity6)

datas_X=[]
for i in range(len(population)-1):
    datas_X.append(population[i]*10000/area[i])
print(datas_X)
def get_lr_stats(yPredict, y, model):
    Residual = sum((y - yPredict) ** 2)  # 残差平方和
    total = sum((y - np.mean(y)) ** 2)  # 总体平方和
    R_square = 1 - Residual / total  # 相关性系数R^2
    message1 = ('相关系数(R^2)： ' + str(R_square))
    return print(message1)


length=len(datas_X)
datas_X=np.array(datas_X).reshape([length, 1])  #将datas_X转化为数组，并变为二维，以符合线性回归拟合函数输入参数要求。

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
model = LinearRegression()
minX = min(datas_X)
maxX = max(datas_X)
X = np.arange(minX,maxX).reshape([-1,1])

colors=["black","red","orange","yellow","green","blue","purple"]
for i in range(len(sortD)):
    linear = linear_model.LinearRegression()           #建立回归方程拟合数据
    sortD[i]=np.array(sortD[i])
    linear.fit(datas_X, sortD[i])
    plt.plot(X, linear.predict(X), color=colors[i])


    # plt.figure(figsize=(9, 9),dpi=500)
    plt.scatter(datas_X, sortD[i], color=colors[i], marker='.', label="事件"+str(i+1))
    plt.plot(datas_X, linear.predict(datas_X),color=colors[i], marker='.', label="事件"+str(i+1))

    plt.xlabel('人口密度')
    plt.ylabel("事件"+str(i+1)+"发生密度")

    y1_pre = linear.predict(datas_X)
    print("事件"+str(i+1)+"线性回归均方差:",mean_squared_error(sortD[i],y1_pre))
    print("事件"+str(i+1)+"线性回归平均绝对误差:",mean_absolute_error(sortD[i],y1_pre))

    print("事件"+str(i+1)+"拟合",end="")
    get_lr_stats(y1_pre,sortD[i] , model)
    print()
    plt.legend()
    plt.show()


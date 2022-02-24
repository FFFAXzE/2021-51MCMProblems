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
    return [pieceA/area[0],pieceB/area[1],pieceC/area[2],pieceD/area[3],pieceE/area[4],pieceF/area[5],pieceG/area[6],pieceH/area[7],pieceI/area[8],pieceJ/area[9],pieceK/area[10],pieceL/area[11],pieceM/area[12],pieceN/area[13],pieceP/area[14]]
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

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
X=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

INF = float('inf')
FloydD=[
    [0, 11.1,INF,11.4,INF,INF,INF,INF,INF,INF,INF,INF,8.2,INF,INF],
    [11.1,0,8.2,12.8,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF],
    [INF,8.2,0,7.7,11.1,9.4,INF,INF,INF,INF,INF,INF,INF,INF,INF],
    [11.4,12.8,7.7,0,INF,6.9,INF,INF,INF,12.7,INF,INF,14.3,10.0,8.5],
    [INF,INF,11.1,INF,0,7.4,INF,INF,INF,INF,INF,INF,INF,INF,INF],
    [INF,INF,9.4,6.9,7.4,0,INF,INF,INF,INF,INF,INF,INF,11.2,INF],
    [INF,INF,INF,INF,INF,INF,0,INF,INF,12.9,13.4,14.5,INF,10.6,INF],
    [INF,INF,INF,INF,INF,INF,INF,0,9.0,INF,INF,12.3,INF,INF,INF],
    [INF,INF,INF,INF,INF,INF,INF,9.0,0,INF,INF,INF,INF,INF,INF],
    [INF,INF,INF,12.7,INF,INF,12.9,INF,INF,0,9.5,INF,9.6,6.9,4.2],
    [INF,INF,INF,INF,INF,INF,13.4,INF,INF,9.5,0,4.4,15.0,INF,INF],
    [INF,INF,INF,INF,INF,INF,14.5,12.3,INF,INF,4.4,0,INF,INF,INF],
    [8.2,INF,INF,14.3,INF,INF,INF,INF,INF,9.6,15.0,INF,0,INF,INF],
    [INF,INF,INF,10.0,INF,11.2,10.6,INF,INF,6.9,INF,INF,INF,0,5.9],
    [INF,INF,INF,8.5,INF,INF,INF,INF,INF,4.2,INF,INF,INF,5.9,0]
]

FloydP=[]
for i in range(len(FloydD)):
    list=[]
    for j in range(len(FloydD)):
        list.append(j)
    FloydP.append(list)

for k in range(len(FloydD)):  #k为中间点
    for i in range(len(FloydD)):  #i为起始点
        for j in range(len(FloydD)):  #j为终点
            if FloydD[i][j]>FloydD[i][k]+FloydD[k][j]:
                FloydD[i][j]=FloydD[i][k]+FloydD[k][j]
                FloydP[i][j] = FloydP[i][k]


N=1
distance=[round(j,N) for j in np.array([i[14] for i in FloydD])]
# distance=[v for v in distance if v!=0]
colors=["black","red","orange","yellow","green","blue","purple"]
# 需要拟合的函数
def f_1(x, A, B):
    return A / x**2
def get_lr_stats(yPredict, y, model):
    Residual = sum((y - yPredict) ** 2)  # 残差平方和
    total = sum((y - np.mean(y)) ** 2)  # 总体平方和
    R_square = 1 - Residual / total  # 相关性系数R^2
    message1 = ('相关系数(R^2)： ' + str(R_square))
    return print(message1)
for t in range(len(sortD)):
    xy=[]
    for i in range(len(distance)):
        xy.append([distance[i],sortD[t][i]])
    xy=np.array(xy)
    xy=xy[np.lexsort(xy[:,::-1].T)]
    # 引用库函数
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import optimize as op

    # 需要拟合的数据组
    x_group = [value[0] for value in xy]
    y_group = [value[1] for value in xy]


    # 得到返回的A，B值
    A, B = op.curve_fit(f_1, x_group, y_group)[0]
    # 数据点与原先的进行画图比较
    plt.scatter(x_group, y_group, color=colors[t],marker='o',label="事件"+str(t+1))
    X=np.arange(0, 40, 1)
    y1 = [A / value ** 2 for value in x_group]
    y=[A/value**2 for value in X]
    y=np.array(y)
    plt.plot(X, y,color=colors[t],label="事件"+str(t+1))

    plt.legend()
    plt.show()
    # print("事件" + str(t + 1) + "拟合均方差:", mean_squared_error(y_group, y1))
    # print("事件" + str(t + 1) + "拟合平均绝对误差:", mean_absolute_error(y_group, y1))

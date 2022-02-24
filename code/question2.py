import datetime
import numpy as np
import pandas as pd
import pylab as mpl
from matplotlib.font_manager import FontProperties
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.metrics import mean_squared_error

fireRescue = pd.read_excel('2.xlsx', sheet_name='Sheet1', header=0)
date = fireRescue['接警日期'].values
date = pd.DatetimeIndex(date)

times=[]
for k in  range(2016,2021):
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
        if date[i].year==k and date[i].month==1:
            mon1+=1
        if date[i].year==k and date[i].month==2:
            mon2+=1
        if date[i].year==k and date[i].month==3:
            mon3+=1
        if date[i].year==k and date[i].month==4:
            mon4+=1
        if date[i].year==k and date[i].month==5:
            mon5+=1
        if date[i].year==k and date[i].month==6:
            mon6+=1
        if date[i].year==k and date[i].month==7:
            mon7+=1
        if date[i].year==k and date[i].month==8:
            mon8+=1
        if date[i].year==k and date[i].month==9:
            mon9+=1
        if date[i].year==k and date[i].month==10:
            mon10+=1
        if date[i].year==k and date[i].month==11:
            mon11+=1
        if date[i].year==k and date[i].month==12:
            mon12+=1
    if k==2020:
        true2020=[mon1,mon2,mon3,mon4,mon5,mon6,mon7,mon8,mon9,mon10,mon11,mon12]
    else:
        times.append([mon1,mon2,mon3,mon4,mon5,mon6,mon7,mon8,mon9,mon10,mon11,mon12])
times=np.array(times)
X=np.array([1,2,3,4,5,6,7,8,9,10,11,12])

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# plt.title('')
plt.plot(X, times[0], color='pink', marker='.',label='2016')
plt.plot(X, times[1], color='green',marker='.',label='2017')
plt.plot(X, times[2], color='blue',marker='.',label='2018')
plt.plot(X, times[3], color='yellow',marker='.',label='2019')
plt.plot(X, true2020, color='black',marker='.',label='2020')
plt.xlabel('月份')
plt.ylabel('出警次数')


predict20=[]
predict21=[]
predict20=times.mean(axis=0)#列
print(predict20)
plt.plot(X, predict20, color='red',marker='*',label='预测2020')


# MSE:均方差
print("预测均方差:",mean_squared_error(true2020,predict20,squared=False))
# MAE:平均绝对误差
print("预测平均绝对误差:",mean_absolute_error(true2020,predict20))
MAPE=0
for i in range(len(predict20)):
    MAPE+=abs(predict20[i]-true2020[i])/true2020[i]
MAPE=MAPE/len(predict20)
print("平均绝对百分比误差：",MAPE)
times=times.tolist()
times.append(true2020)
times=np.array(times)
print(times)
print()
predict21=times.mean(axis=0)#列
print(predict21)
plt.plot(X, predict21, color='purple',marker='*',label='预测2021')
plt.legend()
plt.show()





timedata=pd.date_range('2016-01','2019-12',freq='M')



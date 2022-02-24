import  pandas as pd
import matplotlib.pyplot as plt
from  statsmodels.tsa.arima_model import ARIMA
from  statsmodels.graphics.tsaplots import  plot_acf,plot_pacf
from  statsmodels.tsa.stattools import adfuller as ADF
from  statsmodels.stats.diagnostic import acorr_ljungbox
import datetime
import numpy as np
import pylab as mpl
from matplotlib.font_manager import FontProperties
from pandas import DataFrame


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

firedata=[]
for v1 in times:
   for v2 in v1:
      firedata.append(v2)

firedata=firedata+true2020

timedate=pd.date_range('2016-01-15','2021-01-15',freq='M')
timedate=list(timedate)

# for value in timedate:
#    value=str(value)



data={"日期":timedate,
   "火灾发生次数":firedata}
data=pd.DataFrame(data)
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.plot(data['日期'],data['火灾发生次数'])
plt.title('2016-2020各月火灾发生次数')
plt.show()


diff=data['火灾发生次数'].diff().dropna()
# print(diff)
plt.plot(diff)
plt.show()

acf =plot_acf(diff)
plt.title("acf")
acf.show()

pacf=plot_pacf(diff)
plt.title("pacf")
pacf.show()
print(pacf)


print('差分序列ADF检验结果(小于0.05):',ADF(diff))
print('白噪声检验结果(小于0.05):',acorr_ljungbox(diff,lags=1))

data['火灾发生次数']=data['火灾发生次数'].astype(float)
pmax=int(len(diff)/10)
qmax=int(len(diff)/10)

print()
bic_matrix=[]

for p in range(pmax+1):
   tmp=[]
   for q in range(qmax+1):

      try:  # 存在部分报错，所以用try来跳过报错。
         tmp.append(ARIMA(data['火灾发生次数'], (p, 1, q)).fit().bic)
      except:
         tmp.append(None)

   bic_matrix.append(tmp)
bic_matrix=pd.DataFrame(bic_matrix)
print(bic_matrix)
p,q = bic_matrix.stack().astype('float64').idxmin() #先用stack展平，然后用idxmin找出最小值位置。
print(u'BIC最小的p值和q值为：%s、%s' %(p,q))

result=ARIMA(data['火灾发生次数'], (p, 1, q)).fit()
print("模型报告为：\n",result.summary2())
print("未来一年预测结果、准确误差、置信区间为：\n",result.forecast(12))



firedata=firedata+true2020

timedate2=pd.date_range('2021-01-15','2022-01-15',freq='M')
print(result.forecast(12)[0])
plt.plot(data['日期'],data['火灾发生次数'],color='blue')
plt.plot(timedate2,result.forecast(12)[0],color='red')
plt.title("123")
plt.show()

resid = result.resid #赋值
fig = plt.figure(figsize=(12,8))
fig = plot_acf(resid.values.squeeze(), lags=40)


plt.show()




predict_sunspots = result.predict(start=str('2020-01'),end=str('2022-01'),dynamic=False)
print(predict_sunspots)
fig, ax = plt.subplots(figsize=(12, 8))
ax = sub.plot(ax=ax)
predict_sunspots.plot(ax=ax)
plt.show()

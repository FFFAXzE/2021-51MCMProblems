import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import pylab as mpl

fireRescue = pd.read_excel('2.xlsx', sheet_name='Sheet1', header=0)
time = fireRescue['接警时间点'].values
date = fireRescue['接警日期'].values
date = pd.DatetimeIndex(date)

def everyYear(ye):
    yearsTime=[]
    yearsDate=[]
    years=[]
    for i in range(len(time)):
        if date[i].year == ye:
            yearsDate.append(date[i])
            yearsTime.append(time[i])
    years.append(yearsDate)
    years.append(yearsTime)
    return years

def maxTime(mon,years):
    yearsDate=years[0]
    yearsTime=years[1]
    rel=[]
    list=[]
    list2=[]
    a=[]
    b=[]
    c=[]
    for i in range(len(yearsTime)):
        if yearsDate[i].month == mon or yearsDate[i].month == mon+1 or yearsDate[i].month == (mon+2 if mon+2<=12 else 1):
            list.append(yearsDate[i])
            list2.append(yearsTime[i])
    for k in range(3):

        for j in range(1,32):
            time1 = 0
            time2 = 0
            time3 = 0
            for i in range(len(list)):
                if list[i].month ==(mon+k if mon+k<=12 else 1) and list[i].day==j:

                    if list2[i].hour < 8:
                        time1 += 1
                    elif list2[i].hour < 16:
                        time2 += 1
                    else:
                        time3 += 1
            a.append(time1)
            b.append(time2)
            c.append(time3)
    rel.append(max(a))
    rel.append(max(b))
    rel.append(max(c))
    return rel


def getMaxRescuetime(mon):
    sum2 = []
    maxRescuetime = []
    for i in range(5):
        years=everyYear(2016+i)
        sum2.append(maxTime(mon,years))

    for i in range(3):
        sum=[]
        for j in range(len(sum2)):
            sum.append(sum2[j][i])
        maxRescuetime.append(max(sum))
    return maxRescuetime
maxRescuetime2=getMaxRescuetime(2)
maxRescuetime5=getMaxRescuetime(5)
maxRescuetime8=getMaxRescuetime(8)
maxRescuetime11=getMaxRescuetime(11)
print(maxRescuetime2)
print(maxRescuetime5)
print(maxRescuetime8)
print(maxRescuetime11)
l=[maxRescuetime2,maxRescuetime5,maxRescuetime8,maxRescuetime11]
print(l)

fonts = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=15)
plt.figure(figsize=(9, 9),dpi=170)
mpl.rcParams['font.sans-serif'] = ['FangSong']
col_labels=["月份","时间段","最大出警次数"]

# table_vals = [[2,'0-8',3],[21,22,23],[28,29,30]]
table_vals =[]
for i in range(len(l)):
    for j in range(len(l[i])):
        if j==0:
            table_vals.append([3*i+2,'0-8',l[i][j]])
        elif j==1:
            table_vals.append([3*i+2,'8-16',l[i][j]])
        else:
            table_vals.append([3*i+2,'16-24',l[i][j]])
row_colors = ['red','gold','green']

my_table = plt.table(cellText=table_vals, colWidths=[0.3]*3,colLabels=col_labels, loc='best')

plt.show()

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
areas = fireRescue['事件所在的区域'].values
times=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
areaList=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","P"]
for i in range(len(areas)):
    for j in range(len(times)):
        if areas[i]==areaList[j]:
            times[j]+=1


class Vertex:
    #顶点类
    def __init__(self,vid,outList):
        self.vid = vid#出边
        self.outList = outList#出边指向的顶点id的列表，也可以理解为邻接表
        self.know = False#默认为假
        self.dist = float('inf')#s到该点的距离,默认为无穷大
        self.prev = 0#上一个顶点的id，默认为0
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.vid == other.vid
        else:
            return False
    def __hash__(self):
        return hash(self.vid)

v1=Vertex(1,[2,4,13])
v2=Vertex(2,[1,3,4])
v3=Vertex(3,[2,4,5,6])
v4=Vertex(4,[1,2,3,6,10,13,14,15])
v5=Vertex(5,[3,6])
v6=Vertex(6,[3,4,5,14])
v7=Vertex(7,[10,11,12,14])
v8=Vertex(8,[9,12])
v9=Vertex(9,[8])
v10=Vertex(10,[4,7,11,13,14,15])
v11=Vertex(11,[7,10,12,13])
v12=Vertex(12,[7,8,11])
v13=Vertex(13,[1,4,10,11])
v14=Vertex(14,[4,6,7,10,15])
v15=Vertex(15,[4,10,14])

edges = dict()
def add_edge(front,back,value):
    edges[(front,back)]=value

add_edge(1,2,11.1)
add_edge(1,4,11.4)
add_edge(1,13,8.2)
add_edge(2,1,11.1)
add_edge(2,3,8.2)
add_edge(2,4,12.8)
add_edge(3,2,8.2)
add_edge(3,4,7.7)
add_edge(3,5,11.1)
add_edge(3,6,9.4)
add_edge(4,1,11.4)
add_edge(4,2,12.8)
add_edge(4,3,7.7)
add_edge(4,6,6.9)
add_edge(4,10,12.7)
add_edge(4,13,14.3)
add_edge(4,14,10.0)
add_edge(4,15,8.5)
add_edge(5,3,11.1)
add_edge(5,6,7.4)
add_edge(6,3,9.4)
add_edge(6,4,6.9)
add_edge(6,5,7.4)
add_edge(6,14,11.2)
add_edge(7,10,12.9)
add_edge(7,11,13.4)
add_edge(7,12,14.5)
add_edge(7,14,10.6)
add_edge(8,9,9.0)
add_edge(8,12,12.3)
add_edge(9,8,9.0)
add_edge(10,4,12.7)
add_edge(10,7,12.9)
add_edge(10,11,9.5)
add_edge(10,13,9.6)
add_edge(10,14,6.9)
add_edge(10,15,4.2)
add_edge(11,7,13.4)
add_edge(11,10,9.5)
add_edge(11,12,4.4)
add_edge(11,13,15.0)
add_edge(12,7,14.5)
add_edge(12,8,12.3)
add_edge(12,11,4.4)
add_edge(13,1,8.2)
add_edge(13,4,14.3)
add_edge(13,10,9.6)
add_edge(13,11,15.0)
add_edge(14,4,10.0)
add_edge(14,6,11.2)
add_edge(14,7,10.6)
add_edge(14,10,6.9)
add_edge(14,15,5.9)
add_edge(15,4,8.5)
add_edge(15,10,4.2)
add_edge(15,14,5.9)

#创建一个长度为8的数组，来存储顶点，0索引元素不存
vlist = [False,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15]
#使用set代替优先队列，选择set主要是因为set有方便的remove方法
vset = set([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15])

def getUnknownMin():#此函数则代替优先队列的出队操作
    min = 0
    index = 0
    j = 0
    for i in range(1,len(vlist)):
        boo=vlist[i].know
        vid=vlist[i].dist
        if(boo is True):
            continue
        else:
            if(j==0):
                min = vid
                index = i
            else:
                if(vid < min):
                    min = vid
                    index = i
            j += 1
    #此时已经找到了未知的最小的元素是谁
    vset.remove(vlist[index])#相当于执行出队操作
    return vlist[index]

def getDijkstraResult():
    v15.dist=0
    v10.dist=0
    v14.dist=0
    while(len(vset)!=0):
        v = getUnknownMin()
        v.know = True
        for w in v.outList:
            if(vlist[w].know is True):
                continue
            if(vlist[w].dist == float('inf')):
                vlist[w].dist = v.dist + edges[(v.vid,w)]
                vlist[w].prev = v.vid
            else:
                if((v.dist + edges[(v.vid,w)])<vlist[w].dist):
                    vlist[w].dist = v.dist + edges[(v.vid,w)]
                    vlist[w].prev = v.vid
                else:
                    pass


times=np.array(times)//5
getDijkstraResult()
list = [[v1.prev, v1.dist],
        [v2.prev, v2.dist],
        [v3.prev, v3.dist],
        [v4.prev, v4.dist],
        [v5.prev, v5.dist],
        [v6.prev, v6.dist],
        [v7.prev, v7.dist],
        [v8.prev, v8.dist],
        [v9.prev, v9.dist],
        [v10.prev, v10.dist],
        [v11.prev, v11.dist],
        [v12.prev, v12.dist],
        [v13.prev, v13.dist],
        [v14.prev, v14.dist],
        [v15.prev, v15.dist]]
# for i in range(len(list)):
#     print("v"+str(i+1)+".prev:",list[i][0],"v"+str(i+1)+".dist",list[i][1])
costsum=0
for i in range(len(times)):
    costsum+=times[i]*list[i][1]
print("当前在P区域建消防站，出警总成本为："+str(round(costsum)))


def getDijkstraResult1():
    v15.dist=0
    v10.dist=0
    v14.dist=0
    v5.dist=0
    while(len(vset)!=0):
        v = get_unknown_min()
        v.know = True
        for w in v.outList:
            if(vlist[w].know is True):
                continue
            if(vlist[w].dist == float('inf')):
                vlist[w].dist = v.dist + edges[(v.vid,w)]
                vlist[w].prev = v.vid
            else:
                if((v.dist + edges[(v.vid,w)])<vlist[w].dist):
                    vlist[w].dist = v.dist + edges[(v.vid,w)]
                    vlist[w].prev = v.vid
                else:
                    pass
getDijkstraResult1()
list = [[v1.prev, v1.dist],
        [v2.prev, v2.dist],
        [v3.prev, v3.dist],
        [v4.prev, v4.dist],
        [v5.prev, v5.dist],
        [v6.prev, v6.dist],
        [v7.prev, v7.dist],
        [v8.prev, v8.dist],
        [v9.prev, v9.dist],
        [v10.prev, v10.dist],
        [v11.prev, v11.dist],
        [v12.prev, v12.dist],
        [v13.prev, v13.dist],
        [v14.prev, v14.dist],
        [v15.prev, v15.dist]]
# for i in range(len(list)):
#     print("v"+str(i+1)+".prev:",list[i][0],"v"+str(i+1)+".dist",list[i][1])
costsum1=0
for i in range(len(times)):
    costsum1+=times[i]*list[i][1]
print("2023年"+"在E区域建消防站，出警总成本为："+str(round(costsum1)))


def getDijkstraResult2():
    v15.dist=0
    v10.dist=0
    v14.dist=0
    v5.dist=0
    v1.dist = 0
    while(len(vset)!=0):
        v = get_unknown_min()
        v.know = True
        for w in v.outList:
            if(vlist[w].know is True):
                continue
            if(vlist[w].dist == float('inf')):
                vlist[w].dist = v.dist + edges[(v.vid,w)]
                vlist[w].prev = v.vid
            else:
                if((v.dist + edges[(v.vid,w)])<vlist[w].dist):
                    vlist[w].dist = v.dist + edges[(v.vid,w)]
                    vlist[w].prev = v.vid
                else:
                    pass
getDijkstraResult2()
list = [[v1.prev, v1.dist],
        [v2.prev, v2.dist],
        [v3.prev, v3.dist],
        [v4.prev, v4.dist],
        [v5.prev, v5.dist],
        [v6.prev, v6.dist],
        [v7.prev, v7.dist],
        [v8.prev, v8.dist],
        [v9.prev, v9.dist],
        [v10.prev, v10.dist],
        [v11.prev, v11.dist],
        [v12.prev, v12.dist],
        [v13.prev, v13.dist],
        [v14.prev, v14.dist],
        [v15.prev, v15.dist]]
# for i in range(len(list)):
#     print("v"+str(i+1)+".prev:",list[i][0],"v"+str(i+1)+".dist",list[i][1])
costsum2=0
for i in range(len(times)):
    costsum2+=times[i]*list[i][1]
print("2026年"+"在A区域建消防站，出警总成本为："+str(round(costsum2)))

def getDijkstraResult3():
    v15.dist=0
    v10.dist=0
    v14.dist=0
    v5.dist=0
    v1.dist = 0
    v8.dist = 0
    while(len(vset)!=0):
        v = get_unknown_min()
        v.know = True
        for w in v.outList:
            if(vlist[w].know is True):
                continue
            if(vlist[w].dist == float('inf')):
                vlist[w].dist = v.dist + edges[(v.vid,w)]
                vlist[w].prev = v.vid
            else:
                if((v.dist + edges[(v.vid,w)])<vlist[w].dist):
                    vlist[w].dist = v.dist + edges[(v.vid,w)]
                    vlist[w].prev = v.vid
                else:
                    pass
getDijkstraResult3()
list = [[v1.prev, v1.dist],
        [v2.prev, v2.dist],
        [v3.prev, v3.dist],
        [v4.prev, v4.dist],
        [v5.prev, v5.dist],
        [v6.prev, v6.dist],
        [v7.prev, v7.dist],
        [v8.prev, v8.dist],
        [v9.prev, v9.dist],
        [v10.prev, v10.dist],
        [v11.prev, v11.dist],
        [v12.prev, v12.dist],
        [v13.prev, v13.dist],
        [v14.prev, v14.dist],
        [v15.prev, v15.dist]]
# for i in range(len(list)):
#     print("v"+str(i+1)+".prev:",list[i][0],"v"+str(i+1)+".dist",list[i][1])
costsum3=0
for i in range(len(times)):
    costsum3+=times[i]*list[i][1]
print("2029年"+"在H区域建消防站，出警总成本为："+str(round(costsum3)))










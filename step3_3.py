
#根据lda所得每个学部，每年所得基金数据绘制趋势图
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
labels=['数理科学部','化学科学部','生命科学部','地球科学部','工程与材料科学部','信息科学部','管理科学部','医学科学部','交叉科学部','联合基金','专项项目','应急管理项目']
years=['2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
#读取各学部各年份基金数据

colors = [
    'b','c','r','g','m','y','k','#696969','#FF4500'


]
plt.rcParams['font.sans-serif']=['SimHei']

plt.rcParams['axes.unicode_minus'] = False
def getMoney(areaname,year):
    data_list=[]
    data_path='money_file/'+areaname+'/'
    data_file=year+'_money.txt'
    try:
         with open(data_path+data_file,'r')as f:
            data_lists=f.readlines()
            for item in data_lists:
                data_list.append(item.strip('\n'))
    except:
             print('no file')
    return data_list
def average(data_list):
    lenth=len(data_list)
    sum=0.0
    for i in range(lenth):
        sum=sum+float(data_list[i])

    return sum/lenth
def averagePng(): #绘制各学部每年平均申请项目数
    x=range(2011,2021)
    y_areas=[]
    for i in range(9):
        area_name=labels[i]
        y_area = []
        for j in range(10):
            year=years[j]
            data_list=getMoney(area_name,year)
            y_area.append(average(data_list))
        y_areas.append(y_area)
    # 创建画布
    plt.figure(figsize=(8, 6), dpi=80)  # figsize为横纵坐标比，dpi为像素点个数
    # 绘制曲线
    for i in range(9):
        plt.plot(x, y_areas[i], color=colors[i], linestyle='--', label=labels[i])

    # 绘制曲线图例
    plt.legend(loc='best')  # 提供11种不同的图例显示位置

    # 设置刻度和步长
    z = range(0, 450)
    x_label = ["10:{}".format(i) for i in x]
    plt.xticks(x[::1], x_label[::1])
    plt.yticks(z[::50])
    # 添加网格信息
    plt.grid(linestyle='--', alpha=0.5, linewidth=2)

    # 添加标题
    plt.xlabel('Time/ year')
    plt.ylabel('money/ wanyuan')
    plt.title('project cost')

    # 保存和展示
    plt.savefig('./plt_img/average.png')
    plt.show()
def scatterPng():
    x = range(2011, 2021)
    y_areas = []
    for i in range(9):
        area_name = labels[i]
        y_area = []
        for j in range(10):
            year = years[j]
            data_list = getMoney(area_name, year)
            y_area.append(average(data_list))
        y_areas.append(y_area)
    # 创建画布
    plt.figure(figsize=(8, 6), dpi=80)  # figsize为横纵坐标比，dpi为像素点个数
    markers=[ "o","v","<",">","^","*","h","p","s"]
    for i in range(9):
        for j in range(10):
            plt.scatter(int(years[j]),y_areas[i][j],c=colors[i],marker=markers[i])
    plt.show()
    plt.savefig('./plt_img/scatter.png')
def getNum(areaname):
    data_list=[]
    data_path = 'money_file/' + areaname + '/'
    for year in years:
        data_file = year + '_money.txt'
        try:
            with open(data_path + data_file, 'r') as f:
                data_lists = f.readlines()
                data_list.append(len(data_lists))
        except:
            print('no file')

    return data_list
def boxPlot():
    rcParams['axes.unicode_minus'] = False
    rcParams['font.sans-serif'] = ['Simhei']
    data_list=[]
    pro=[]
    for i in range(9):
        area_name =labels[i]
        data_list.append(getNum(area_name))
        pro.append(area_name)
    plt.boxplot(data_list, notch=False, labels=pro, patch_artist=False,
                boxprops={'color': 'black', 'linewidth': '2.0'},
                capprops={'color': 'black', 'linewidth': '2.0'})
    plt.xlabel("学部", fontsize=20)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=20)
    plt.savefig('plt_img/box.png')
    plt.show()
boxPlot()




#averagePng()
#scatterPng()

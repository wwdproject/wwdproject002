
#根据lda所得每个学部，每年所得基金数据绘制趋势图
import matplotlib.pyplot as plt
import random
labels=['数理科学部','化学科学部','生命科学部','地球科学部','工程与材料科学部','信息科学部','管理科学部','医学科学部','交叉科学部','联合基金','专项项目','应急管理项目']
years=['2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
#读取各学部各年份基金数据

colors = [
    'b','c','r','g','m','y','k','#696969','#FF4500'


]
plt.rcParams['font.sans-serif']=['SimHei']

plt.rcParams['axes.unicode_minus'] = False
def getPronum(areaname,year):
    data_list=[]
    data_path='money_file/'+areaname+'/'
    data_file=year+'_money.txt'
    try:
         with open(data_path+data_file,'r')as f:
            data_lists=f.readlines()
            return len(data_lists)
    except:
             print('no file')
    return 0
def average(data_list):
    lenth=len(data_list)
    sum=0.0
    for i in range(lenth):
        sum=sum+float(data_list[i])

    return sum/lenth
def ProPng(): #绘制各学部每年平均基金
    x=range(2011,2021)
    y_areas=[]
    for i in range(9):
        area_name=labels[i]
        y_area = []
        for j in range(10):
            year=years[j]
            num=getPronum(area_name,year)
            y_area.append(num)
        y_areas.append(y_area)
    # 创建画布
    plt.figure(figsize=(8, 6), dpi=80)  # figsize为横纵坐标比，dpi为像素点个数
    # 绘制曲线
    for i in range(9):
        plt.plot(x, y_areas[i], color=colors[i], linestyle='--', label=labels[i])

    # 绘制曲线图例
    plt.legend(loc='best')  # 提供11种不同的图例显示位置

    # 设置刻度和步长
    z = range(0, 100)
    x_label = ["10:{}".format(i) for i in x]
    plt.xticks(x[::1], x_label[::1])
    plt.yticks(z[::10])
    # 添加网格信息
    plt.grid(linestyle='--', alpha=0.5, linewidth=2)

    # 添加标题
    plt.xlabel('Time/ year')
    plt.ylabel('num')
    plt.title('project num')

    # 保存和展示
    plt.savefig('./plt_img/Pro.png')
    plt.show()
'''
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
'''



ProPng()
#scatterPng()

# 多曲线图
x = range(60)
# print(type(x))
y1 = [random.uniform(35, 40) for i in x]  # uniform提供[35, 40)随机值
y2 = [random.uniform(25, 30) for j in x]
# print(type(y))

# 创建画布
plt.figure(figsize=(8, 6), dpi=80)  # figsize为横纵坐标比，dpi为像素点个数

# 绘制曲线
plt.plot(x, y1, color='r', linestyle='--', label='Shanghai')
plt.plot(x, y2, color='g', linestyle='-.', label='Beijing')

# 绘制曲线图例
plt.legend(loc='best')  # 提供11种不同的图例显示位置

# 设置刻度和步长
z = range(-10, 45)
x_label = ["10:{}".format(i) for i in x]
plt.xticks(x[::5], x_label[::5])
plt.yticks(z[::5])

# 添加网格信息
plt.grid(linestyle='--', alpha=0.5, linewidth=2)

# 添加标题
plt.xlabel('Time/ min')
plt.ylabel('Temperature/ ℃')
plt.title('Curve of Temperature Change with Time')

# 保存和展示
# plt.savefig('./plt_img/test2.png')
plt.show()
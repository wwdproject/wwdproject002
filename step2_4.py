#分析各学部主题词的相似性
import numpy as np
import re

import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") #过滤掉警告的意思
#from pyforest import *
from matplotlib import pyplot as plt
#图片显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False #减号unicode编码

labels=['数理科学部','化学科学部','生命科学部','地球科学部','工程与材料科学部','信息科学部','管理科学部','医学科学部','交叉科学部','联合基金','专项项目','应急管理项目']
years=['2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
topic_area=[]
word_vec=[]
top_count=15
'''
>>> import re
>>> string  = "本实用新型公开了全       自动套鞋机，在机架（1）内安装鞋套弹性进给装置构成套鞋机，在机架（1）的中间安装滑杆（2）"
>>> res= re.sub(u"\\（.*?\\）|\\{.*?}|\\［.*?］", '', string  )# 全角字符
>>> print(res)
本实用新型公开了全       自动套鞋机，在机架内安装鞋套弹性进给装置构成套鞋机，在机架的中间安装滑杆
>>>
'''
def simlarity(data_list1,data_list2):
    p=len(data_list1) #数据的维数

    s=0.0
    for i in range(p):
        s+=(data_list1[i]-data_list2[i])**2

    return s
def getVec(areaname,year):
    #返回该学部该学年主题词向量[[],[]]
    wordvec_path='lda_all_result/topic_emb/'+areaname+'/'
    wordvec_file=year+'_lda_word_emb.txt'
    try:
        with open(wordvec_path + wordvec_file, 'r') as f:
            txt_lists = f.readlines()
            word_vec = []
            word_vec_item = []
            i = 0
            for txt_list in txt_lists:
                txt_list = txt_list.split()
                for txt_list_item in txt_list:
                    if (txt_list_item == '[' or txt_list_item == ']'):
                        continue
                    txt_list_item = txt_list_item.strip(']')
                    txt_list_item = txt_list_item.strip('[')
                    txt_list_item = txt_list_item.strip(']')
                    txt_list_item = txt_list_item.strip('[')

                    word_vec_item.append(float(txt_list_item))
                    i = i + 1
                    print(i)
                    if (i == 100):
                        word_vec.append(word_vec_item)

                        word_vec_item = []
                        i = 0
    except:
        word_vec=[]


    return word_vec

def QuickSort(nums,left,right):
    if left>=right:
        return 0
    baseNumber=nums[left]
    i=left
    j=right
    while i!=j:
        while j>i and nums[j]>=baseNumber:
            j-=1
        if j>i:
        #实现两个元素的互换，python就是简单啊
            nums[i],nums[j]=nums[j],nums[i]

        while i<j and nums[i]<=baseNumber:
            i+=1
        if i<j:
            nums[i],nums[j]=nums[j],nums[i]
    QuickSort(nums,left,i-1)
    QuickSort(nums,i+1,right)

def compare(word_area1,word_area2):
    #比较两个学部主题词相似度
    sim_resut=[]
    for i in range(len(word_area1)):
        s=9999
        for j in range(len(word_area2)):
            if(i==j):
                continue
            t=simlarity(word_area1[i],word_area2[j])
            if(s>t):
                s=t

        sim_resut.append(s)
    print(sim_resut)
    QuickSort(sim_resut,0,len(sim_resut)-1)
    print(sim_resut)
    sim_num=0.0
    for i in range(min(top_count,len(word_area1),len(word_area2))):

        if(sim_resut[i] is not None):
            sim_num+=sim_resut[i]
        else:
            continue


    return sim_num/min(top_count,len(word_area1),len(word_area2))




def makePng():
    #所有学部所有学年主题词的矩阵

    word_vec = []
    area=[]
    for i in range(9):
        area_name=labels[i]
        area.append(area_name)
        #该学部所有学年的主题词的矩阵
        word_area_vec=[]
        for j in range(10):
            year=years[j]
            word_area_vec=word_area_vec+getVec(area_name,year)
        word_vec.append(word_area_vec)
    #创建用来存储两学部该学年所有主题词相似度的矩阵,9*9
    sim_sum=[]
    for i in range(9):
        sim_item=[]
        for j in range(9):
            sim_item.append([0])
        sim_sum.append(sim_item)
    for i in range(9):
        word_area1=word_vec[i]
        for j in range(9):
            word_area2=word_vec[j]
            sim_sum[i][j]=compare(word_area1,word_area2)
    print(sim_sum)

    ax = plt.subplots(figsize=(50, 50))  # 调整画布大小
    ax = sns.heatmap(sim_sum, vmax=.01, square=True, annot=True)  # 画热力图   annot=True 表示显示系数
    ax.set_yticklabels(area, fontsize=20, rotation=360, horizontalalignment='right')
    ax.set_xticklabels(area, fontsize=20, horizontalalignment='right')
    # 设置刻度字体大小
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig("plt_img/heat2.png")



makePng()
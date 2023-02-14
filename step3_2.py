
from gensim import models
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import models
from gensim.models import FastText  # 使用FastText
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wordcloud
from imageio import imread
topic_count = 8  # lda模型主题数量
topn_word = 8  # 每个主题选取的词数
ids=['1','2','3','4','5','6','7','8','9','U','J','L']
labels=['数理科学部','化学科学部','生命科学部','地球科学部','工程与材料科学部','信息科学部','管理科学部','医学科学部','交叉科学部','联合基金','专项项目','应急管理项目']
id_num=[0,1,2,3,4,5,6,7,8,9,10,11]
dropwords=[]
with open('停用词','r',encoding='utf-8') as f:
    words=f.readlines()
    for word in words:
        word=word.strip('\n')
        dropwords.append(word)
def getWordbyArea(areaname): #按照所属学部获取保存的分词
    word_lists1=[]  #只经过去除标点的分词
    word_lists2=[]  #去除过标点和停用词的分词
    word_file_path='word_file/'+'word_savedbyarea/'+areaname+'/'
    years=['2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
    print('begin')
    for year in years:
        try:

            with open(word_file_path+year+'_1.csv','r',encoding='utf-8') as f1:
                txts_lists = f1.readlines()
                for txts_list in txts_lists:
                    word_list = []
                    txts_list = txts_list.strip('\n')
                    txts_list = txts_list.strip('[')
                    txts_list = txts_list.strip(']')
                    txts_list = txts_list.split(',')
                    for word in txts_list:
                        word = word.replace('\'', '').strip(' ')
                        if (word is not None):
                            word_list.append(word)
                    word_lists1.append(word_list)
        except:
            print('err')
            continue
        try:

            with open(word_file_path+year+'_2.csv','r',encoding='utf-8') as f2:
                txts_lists = f2.readlines()
                for txts_list in txts_lists:
                    word_list = []
                    txts_list = txts_list.strip('\n')
                    txts_list = txts_list.strip('[')
                    txts_list = txts_list.strip(']')
                    txts_list = txts_list.split(',')
                    for word in txts_list:
                        word = word.replace('\'', '').strip(' ')
                        if (word is not None):
                            word_list.append(word)
                    word_lists2.append(word_list)
        except:
            print('err')
            continue
    return word_lists1,word_lists2
def getTopicbyArea(areaname):
    lda_model_path='lda_ap_result/models/lda/'
    lda_model_file=areaname+'_lda.model'
    lda_model = models.ldamodel.LdaModel.load(lda_model_path + lda_model_file)
    lda_word_list = []

    for i in range(topic_count):
        lda_word = []
        for j in range(topn_word):
            lda_word.append(lda_model.show_topic(i, topic_count)[j][0])  # 获取单词矩阵
        lda_word_list=lda_word_list+lda_word
    lda_word_list=list(set(lda_word_list))
    return lda_word_list
for i in range(9):

    mask_image = imread("python.png")
    w = wordcloud.WordCloud(
        width=800,
        height=800,  # 设置图片长宽为1080p
        mask=mask_image,
        background_color='white',
        font_path='C://Windows//Fonts/msyh.ttc',  # 设置字体为微软雅黑
        max_words=50,  # 设置词汇最大数量为300
        colormap='magma',  # 设置配色集为magma
        collocations=False
    )
    area_name=labels[i]
    word_list1,word_list2=getWordbyArea(area_name)
    topic_word=getTopicbyArea(area_name)
    txts = ''
    for topic in topic_word:
        if(topic in dropwords):
            continue
        for word_list in word_list2:
            for word in word_list:
                if (word ==topic):
                    txts = txts + ' ' + word
    w.generate(txts)
    topic_path='lda_all_result/topic_png/'
    topic_file=area_name+'_topic.png'
    w.to_file(topic_path+topic_file)
    print(area_name+'saved success')

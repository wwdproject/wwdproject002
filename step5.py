
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
topic_count = 100  # lda模型主题数量
topn_word = 30  # 每个主题选取的词数
topic_path='topic/'
result_path = '2001_2020_lda10030_ap_result/'  # 输出结果及模型保存路径
lda_model_path = result_path  # 保存lda模型的目录
lda_model_file = '2001_2020lda.model'  # lda模型保存文件名
word_lists2=[]
#加载原分词
with open('words_2.csv','r',encoding='utf-8')as f:
    txts_lists=f.readlines()
    for txts_list in txts_lists:
        word_list=[]
        txts_list=txts_list.strip('\n')
        txts_list=txts_list.strip('[')
        txts_list = txts_list.strip(']')
        txts_list = txts_list.split(',')
        for word in txts_list:
            word =word.replace('\'','').strip(' ')
            if(word is not None):
                word_list.append(word)
        word_lists2.append(word_list)
lda_model = models.ldamodel.LdaModel.load(lda_model_path + lda_model_file)
w = wordcloud.WordCloud(
    width=1920,
    height=1080,  # 设置图片长宽为1080p
    background_color='white',  # 设置背景颜色为白色
    font_path='C://Windows//Fonts/msyh.ttc',  # 设置字体为微软雅黑
    max_words=30,  # 设置词汇最大数量为300
    colormap='magma',  # 设置配色集为magma
    collocations = False
)
#加载处理后的主题词
lda_word_list = []
lda_weight_list = []
for i in range(topic_count):
    lda_word = []
    lda_weight = []
    for j in range(topn_word):
        lda_word.append(lda_model.show_topic(i, topic_count)[j][0])  # 获取单词矩阵
        lda_weight.append(lda_model.show_topic(i, topic_count)[j][1])  # 获取值矩阵
    lda_word_list.append(lda_word)
    lda_weight_list.append(lda_weight)
topic_words=[]
i=0
for lda_word in lda_word_list:
    txts=''
    for word_list in word_lists2:
        for word in word_list:
            if(word in lda_word):
                txts=txts+' '+word
    w.generate(txts)
    w.to_file(topic_path+"test"+str(i)+"topic.png")
    i=i+1
    print(i)
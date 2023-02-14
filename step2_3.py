import jieba
import numpy as np
import pandas as pd
import pyLDAvis
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
import string
from zhon.hanzi import punctuation
import wordcloud
from imageio import imread
import pyLDAvis.gensim as gensimvis
import warnings

warnings.filterwarnings('ignore')
drop_below = 1  # 删除低频词->至少在i个条目中出现
drop_above = 0.5  # 删除高频词->在i%及以上的条目中出现
topic_count = 9  # lda模型主题数量
topn_word = 9 # 每个主题选取的词数
labels=['数理科学部','化学科学部','生命科学部','地球科学部','工程与材料科学部','信息科学部','管理科学部','医学科学部','交叉科学部','联合基金','专项项目','应急管理项目']
years=['2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
punctions=punctuation
punctions=punctions+string.punctuation
chinese_punctions=[]
for i in range(len(punctions)):
    chinese_punctions.append(punctions[i])
dropwords=[]
with open('停用词','r',encoding='utf-8') as f:
    words=f.readlines()
    for word in words:
        word=word.strip('\n')
        dropwords.append(word)
def getWordbyArea(areaname,year): #按照所属学部,年份获取保存的分词
    word_lists1=[]  #只经过去除标点的分词
    word_lists2=[]  #去除过标点和停用词的分词
    word_file_path='word_file/'+'word_savedbyarea/'+areaname+'/'
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
            return None

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
            print('not find file')

    return word_lists1,word_lists2
def lda_by_area(areaname,year,corpus, dictionary): #创建各学部,各年份的lda
    lda_model_path='lda_all_result/models/lda/'+areaname+'_lda/'
    lda_model_file=year+'_lda.model'
    #print(dictionary)
    '''
     try:
        lda_model = models.ldamodel.LdaModel.load(lda_model_path + lda_model_file)
        print('读取lda模型成功')
    except:
        if(dictionary is None):
            print(areaname)
        
    
    
    '''
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=topic_count,
        passes=120,
        update_every=0,
        alpha='auto',
        iterations=500
    )
    lda_model.save(lda_model_path + lda_model_file)  # 保存模型
    #print('lda模型构建成功')
    lda_word_list = []
    lda_weight_list = []
    lda_final_list = []
    topic_word=[]
    for i in range(topic_count):
        lda_word = []
        lda_weight = []
        for j in range(topn_word):
            if(lda_model.show_topic(i, topic_count)[j][0] is None):
                continue
            lda_word.append(lda_model.show_topic(i, topic_count)[j][0])  # 获取单词矩阵
            lda_weight.append(lda_model.show_topic(i, topic_count)[j][1])  # 获取值矩阵
            topic_word=topic_word+lda_word

        lda_word_list.append(lda_word)
        lda_weight_list.append(lda_weight)
    

    lda_final_list = [lda_word_list, lda_weight_list]
    lda_result_path='lda_all_result/lda_result/'+areaname+'_result/'
    lda_result_file=year+'_result.txt'

    with open(lda_result_path + lda_result_file, 'w') as f:
        for list in lda_word_list:
            f.write(str(list))

    return lda_final_list,lda_word_list, lda_weight_list,topic_word
def embedding(areaname,year,fasttext_model, word_need):
    # word_list是一个二维列表[[句子1]，[句子2].......]
    # word_need是一个二维列表[[词1，词2...]，[词1，词2...]...]
    '''
    sentences = word_list
    fasttext_path='lda_all_result/models/fasttext/'+areaname+'_fasttext'
    fasttext_file=year+'_fasttext.model'
    try:
        fasttext_model = models.fasttext.FastText.load(fasttext_path + fasttext_file)
        print('读取fasttext词嵌入模型成功')
    except:
        fasttext_model = FastText(
            sentences,
            vector_size=50,
            min_count=1
        )
        fasttext_model.save(fasttext_path + fasttext_file)
    :param areaname:
    :param year:
    :param fasstext_model:
    :param word_need:
    :return:
    '''


    word_vec_matrix = [[fasttext_model.wv[word] for word in words] for words in word_need]
    # 获取词矩阵对应的嵌入向量三维矩阵[主题[词[词向量元素]]]
    lda_word_emb_path='lda_all_result/word_emb/'+areaname+'/'
    lda_word_emb_file=year+'_lda_word_emb.txt'
    with open(lda_word_emb_path + lda_word_emb_file, 'w') as f:
        word_vec_matrix_listtosave = np.array(word_vec_matrix).tolist()
        f.write(str(word_vec_matrix_listtosave))
   # print('word_vec_matrix的维度' + str(np.array(word_vec_matrix_listtosave).shape))
    return word_vec_matrix


def ap(areaname,year,word_vec_matrix, weight_matrix):
    nor_weight_matrrix = np.zeros_like(weight_matrix)
    for i in range(topic_count):
        for j in range(topn_word):
            nor_weight_matrrix[i][j] = weight_matrix[i][j] / sum(weight_matrix[i])
    #这是一个二维矩阵，是weight_matrix矩阵每行归一化处理的结果
    weight_word_matrix = np.zeros_like(word_vec_matrix)
    for i in range(topic_count):
        for j in range(topn_word):
            weight_word_matrix[i][j] = word_vec_matrix[i][j] * nor_weight_matrrix[i][j]
            break
    # 获取加权后的词嵌入向量三维矩阵
    #最低维是一个ndarray,是多维向量乘以权
    #NumPy提供了一个N维数组类型，即ndarray， 它描述了相同类型的“项目”集合。可以使用例如N个整数来索引项目。从数组中提取的项（ 例如 ，通过索引）
    # 由Python对象表示， 其类型是在NumPy中构建的数组标量类型之一。 数组标量允许容易地操纵更复杂的数据排列。
    sum_matrix = np.zeros([topic_count, np.array(word_vec_matrix).shape[2]])
    for i in range(topic_count):
        for j in range(np.array(word_vec_matrix).shape[2]):
            for t in range(topn_word):
                sum_matrix[i][j] = sum_matrix[i][j] + weight_word_matrix[i][t][j]
    # 获取压缩后的主题向量矩阵(topic_count * dim)
    lda_topic_emb_path = 'lda_all_result/topic_emb/'+areaname+'/'
    lda_topic_emb_file =year+ '_lda_word_emb.txt'
    with open(lda_topic_emb_path + lda_topic_emb_file, 'w') as f:
        #f.write(str(sum_matrix.tolist()))
        for i in range(topic_count):
            for j in range(topn_word):
                f.write(str(word_vec_matrix[i][j]))
                f.write(' ')


    ap = AffinityPropagation(
        affinity='euclidean',  # 尝试欧氏距离
        damping=0.5, max_iter=200, convergence_iter=30, preference=-50

    ).fit(sum_matrix)
    if(areaname=='信息科学部'):
        print(ap.labels_)

    return ap.cluster_centers_indices_, ap.labels_, sum_matrix  # 中心主题词的索引,标签

def lda(): #构建每个学部，各年份的lda模型
    #先获取所有分词数据，构建词向量
    word_all_list=[]
    fasttext_path = 'lda_all_result/models/fasttext/'  + 'all_fasttext/'
    fasttext_file =  '_fasttext.model'

    for i in range(9):
        for j in range(10):
            area_name = labels[i]
            year = years[j]
            word_file_path = 'word_file/' + 'word_savedbyarea/' + area_name + '/'
            word_file_name = year + '_1.csv'
            word_list1, word_list2 = getWordbyArea(area_name, year)
            word_all_list=word_all_list+word_list1

    '''
        try:
        fasttext_model = models.fasttext.FastText.load(fasttext_path + fasttext_file)
        print('读取fasttext词嵌入模型成功')
    except:
        
    
    '''

    fasttext_model = FastText(
        word_all_list,
        vector_size=100,
        min_count=1
    )
    fasttext_model.save(fasttext_path + fasttext_file)
    for i in range(9):
        for j in range(10):
            area_name=labels[i]
            year=years[j]
            word_file_path='word_file/'+'word_savedbyarea/'+area_name+'/'
            word_file_name=year+'_1.csv'
            word_list1, word_list2 = getWordbyArea(area_name, year)
            if(i==5):
                print(word_file_path+word_file_name+'first')
            dictionary = Dictionary(word_list2)  # 构建gensim字典，包含词编号，词频，词名
            dictionary.filter_extremes(no_below=drop_below, no_above=drop_above)  # 过滤高低频词
            if(dictionary is None or len(dictionary)==0):
                print('err'+word_file_path+word_file_name)
                print(word_list2)
                continue
            corpus = [dictionary.doc2bow(text) for text in word_list2]
            tfidf_path = 'lda_all_result/models/tf_idf/' + area_name + '_tfidf/'
            tfidf_file = year + '_tfidf'
            '''
            try:
                tfidf = models.tfidfmodel.TfidfModel.load(tfidf_path + tfidf_file)
                print('读取tfidf模型成功')
            except:
                tfidf = models.TfidfModel(corpus)
                tfidf.save(tfidf_path + tfidf_file)
                print('tfidf模型构建完成')
            
            '''
            tfidf = models.TfidfModel(corpus)
            tfidf.save(tfidf_path + tfidf_file)

            corpus_tfidf = tfidf[corpus]
            if (i == 5):
                print(word_file_path + word_file_name+'second')
            lda_list, lda_word_list, lda_weight_list,topic_word = lda_by_area(area_name, year, corpus_tfidf, dictionary)
            if (i == 5):
                print(word_file_path + word_file_name+'third')

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
            txts = ''
            for topic in topic_word:
                if (topic in dropwords):
                    continue
                for word_list in word_list2:
                    for word in word_list:
                        if (word == topic):
                            txts = txts + ' ' + word
            w.generate(txts)
            topic_path = 'lda_all_result/topic_png/'
            topic_file = area_name+'_'+year + '_topic.png'
            w.to_file(topic_path + topic_file)
            print(area_name+year + 'saved success')









            word_vec_matrix = embedding(area_name,year,fasttext_model, lda_list[0])  # 词向量获取，使用未过滤停用词的语料
            center_topic_index, center_labels, ap_fit_x = ap(area_name,year, word_vec_matrix, lda_list[1])
            center_topic_id = 1
            dic = {}
            topic_list=[]
            for id in center_topic_index:
                dic['TopicWord:' + str(center_topic_id)] = lda_list[0][id]
                dic['TopicWeight:' + str(center_topic_id)] = lda_list[1][id]
                topic_list.append(lda_list[0][id])
                center_topic_id = center_topic_id + 1
            center_word_df = pd.DataFrame(dic)
            if (i == 5):
                print(word_file_path + word_file_name+'4')
            final_result_path = 'lda_all_result/final_result/'+area_name+'/'
            final_result_file = year + '_final_result.csv'
            #print(dic)
            print(i,j)
            center_word_df.to_csv(final_result_path + final_result_file, sep=',')
            with open(final_result_path+year+'_finaltopic.csv','w')as f:
                if (i == 5):
                    print(word_file_path + word_file_name+'5')
                for item in topic_list:
                    for item_item in item:
                        f.write(item_item)
                        f.write('\n')





lda()

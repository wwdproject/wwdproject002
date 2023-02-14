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
result_path='lda_ap_result/'
type_area='type_area/'
lda_result_path = result_path  # lda结果存放路径
lda_result_file = '2001_2020lda_result.txt'  # lda结果保存文件名
fasttext_path = result_path  # fasttext模型路径
fasttext_file = 'fasttext.model'  # fasttext模型保存文件名
word_list_path = result_path  # 词表保存路径
word_list_file = 'wordlist2001_2020.txt'  # 词表保存文件名
tfidf_path = result_path  # tfidf保存路径
tfidf_file = 'tfidf'  # tfidf保存文件名
lda_model_path = result_path +'lda/' # 保存lda模型的目录
lda_model_file = 'lda.model'  # lda模型保存文件名
lda_topic_emb_path = result_path
lda_topic_emb_file = 'lda_topic_emb.txt'  # 保存主题向量矩阵
lda_word_emb_path = result_path
lda_word_emb_file = 'lda_word_emb.txt'
final_result_path = result_path
final_result_file = '2001_2020final_result.csv'
img_result_path = result_path
img_result_file = 'img_result.png'
dropwords=[]
with open('停用词','r',encoding='utf-8') as f:
    words=f.readlines()
    for word in words:
        word=word.strip('\n')
        dropwords.append(word)
drop_below = 5  # 删除低频词->至少在i个条目中出现
drop_above = 0.5  # 删除高频词->在i%及以上的条目中出现
topic_count = 8  # lda模型主题数量
topn_word = 5  # 每个主题选取的词数
colors = [
    '#F0F8FF','#FAEBD7','#00FFFF','#7FFFD4','#F0FFFF','#F5F5DC','#FFE4C4','#000000','#FFEBCD','#0000FF','#8A2BE2','#A52A2A','#DEB887','#5F9EA0','#7FFF00','#D2691E','#FF7F50','#6495ED','#FFF8DC',
    '#DC143C','#00FFFF','#00008B','#008B8B','#B8860B','#A9A9A9','#006400','#BDB76B','#8B008B','#556B2F','#FF8C00','#9932CC','#8B0000','#E9967A','#8FBC8F','#483D8B','#2F4F4F','#00CED1','#9400D3',
    '#FF1493','#00BFFF','#696969','#1E90FF','#B22222','#FFFAF0','#228B22','#FF00FF','#DCDCDC','#F8F8FF','#FFD700','#DAA520','#808080','#008000','#ADFF2F','#F0FFF0','#FF69B4','#CD5C5C','#4B0082',
    '#FFFFF0','#F0E68C','#E6E6FA','#FFF0F5','#7CFC00','#FFFACD','#ADD8E6','#F08080','#E0FFFF','#FAFAD2','#90EE90','#D3D3D3','#FFB6C1','#FFA07A','#20B2AA','#87CEFA','#778899','#B0C4DE','#FFFFE0',
    '#00FF00','#32CD32','#FAF0E6','#FF00FF','#800000','#66CDAA','#0000CD','#BA55D3','#9370DB','#3CB371','#7B68EE','#00FA9A','#48D1CC','#C71585','#191970','#F5FFFA','#FFE4E1','#FFE4B5','#FFDEAD',
    '#000080','#FDF5E6','#808000','#6B8E23','#FFA500','#FF4500','#DA70D6','#EEE8AA','#98FB98','#AFEEEE','#DB7093','#FFEFD5','#FFDAB9','#CD853F','#FFC0CB','#DDA0DD','#B0E0E6','#800080','#FF0000',
    '#BC8F8F','#4169E1','#8B4513','#FA8072','#FAA460','#2E8B57','#FFF5EE','#A0522D','#C0C0C0','#87CEEB','#6A5ACD','#708090','#FFFAFA','#00FF7F','#4682B4','#D2B48C','#008080','#D8BFD8','#FF6347',
    '#40E0D0','#EE82EE','#F5DEB3','#FFFFFF','#F5F5F5','#FFFF00','#9ACD32'
]
ids=['1','2','3','4','5','6','7','8','9','U','J','L']
years=['2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
labels=['数理科学部','化学科学部','生命科学部','地球科学部','工程与材料科学部','信息科学部','管理科学部','医学科学部','交叉科学部','联合基金','专项项目','应急管理项目']
id_num=[0,1,2,3,4,5,6,7,8,9,10,11]
def lda_by_year(year,corpus, dictionary): #创建各学部的lda
    lda_model_path='lda_all_result/models/lda/'+year+'/'
    lda_model_file='lda.model'
    try:
        lda_model = models.ldamodel.LdaModel.load(lda_model_path + lda_model_file)
        print('读取lda模型成功')
    except:
        if(dictionary is None):
            print(year)
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=topic_count,
            passes=2,
            update_every=0,
            alpha='auto',
            iterations=500
        )
        lda_model.save(lda_model_path + lda_model_file)  # 保存模型

        print('lda模型构建成功')

    lda_word_list = []
    lda_weight_list = []
    lda_final_list = []
    topic_word=[]
    for i in range(topic_count):
        lda_word = []
        lda_weight = []

        for j in range(topn_word):
            lda_word.append(lda_model.show_topic(i, topic_count)[j][0])  # 获取单词矩阵
            lda_weight.append(lda_model.show_topic(i, topic_count)[j][1])  # 获取值矩阵
        lda_word_list.append(lda_word)
        topic_word=topic_word+lda_word
        lda_weight_list.append(lda_weight)
    lda_final_list = [lda_word_list, lda_weight_list]
    lda_result_path='lda_all_result/lda_result/'+year+'/'
    lda_result_file='result.txt'
    with open(lda_result_path + lda_result_file, 'w') as f:
        f.write(str(lda_final_list))
    # 写入格式为[[[]],[[]]]
    return lda_final_list,topic_word
def getWordbyTime(year): #按照所属学部获取保存的分词
    word_lists1=[]  #只经过去除标点的分词
    word_lists2=[]  #去除过标点和停用词的分词
    word_file_path='word_file/'+'word_savedbyyear/'+year+'/'
    for label in labels:
        try:

            with open(word_file_path+label+'_1.csv','r',encoding='utf-8') as f1:
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

            with open(word_file_path+label+'_2.csv','r',encoding='utf-8') as f2:
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


def embedding(year,word_list, word_need):
    # word_list是一个二维列表[[句子1]，[句子2].......]
    # word_need是一个二维列表[[词1，词2...]，[词1，词2...]...]
    sentences = word_list
    fasttext_path='lda_all_result/models/fasttext/'+year+'/'
    fasttext_file='fasttext.model'
    try:
        fasttext_model = models.fasttext.FastText.load(fasttext_path + fasttext_file)
        print('读取fasttext词嵌入模型成功')
    except:
        fasttext_model = FastText(
            sentences,
            vector_size=100,
            min_count=1
        )
        fasttext_model.save(fasttext_path + fasttext_file)

    word_vec_matrix = [[fasttext_model.wv[word] for word in words] for words in word_need]
    # 获取词矩阵对应的嵌入向量三维矩阵[主题[词[词向量元素]]]
    lda_word_emb_path='lda_all_result/word_emb/'+year+'/'
    lda_word_emb_file='lda_word_emb.txt'
    with open(lda_word_emb_path + lda_word_emb_file, 'w') as f:
        word_vec_matrix_listtosave = np.array(word_vec_matrix).tolist()
        f.write(str(word_vec_matrix_listtosave))
    print('word_vec_matrix的维度' + str(np.array(word_vec_matrix_listtosave).shape))
    return word_vec_matrix
def ap(year,word_vec_matrix, weight_matrix):
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
    lda_topic_emb_path = 'lda_all_result/topic_emb/'+year+'/'
    lda_topic_emb_file =  'lda_word_emb.txt'
    with open(lda_topic_emb_path + lda_topic_emb_file, 'w') as f:
        f.write(str(sum_matrix.tolist()))

    ap = AffinityPropagation(
        affinity='euclidean',  # 尝试欧氏距离
    ).fit(sum_matrix)
    return ap.cluster_centers_indices_, ap.labels_, sum_matrix  # 中心主题词的索引,标签

def lda(): #输出lda模型
    for i in range(10):
        year=years[i]
        word_list1,word_list2=getWordbyTime(year)
        # 构建字典与词袋
        dictionary = Dictionary(word_list2)  # 构建gensim字典，包含词编号，词频，词名
        dictionary.filter_extremes(no_below=drop_below, no_above=drop_above)  # 过滤高低频词
        corpus = [dictionary.doc2bow(text) for text in word_list2]

        # 第三步：构建tfidf模型
        # 一种用于资讯检索与资讯探勘的常用加权技术。TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度
        # 单词的重要性随着它在文档中出现的次数而增加，随着它在所有文档中出现的频率而下降。
        # 参考文献：http://www.ruanyifeng.com/blog/2013/03/tf-idf.html
        tfidf_path='lda_all_result/models/tf_idf/'+year+'/'
        tfidf_file='tfidf'
        try:
            tfidf = models.tfidfmodel.TfidfModel.load(tfidf_path + tfidf_file)
            print('读取tfidf模型成功')
        except:
            tfidf = models.TfidfModel(corpus)
            tfidf.save(tfidf_path + tfidf_file)
            print('tfidf模型构建完成')
        corpus_tfidf = tfidf[corpus]
        lda_list,topic_word = lda_by_year(year,corpus_tfidf, dictionary)
        '''
            构建lda模型
            获取一个列表：
            [[单词矩阵],[值矩阵]] -> topic_count * topn_word * 2
        '''
        mask_image = imread("python.png")
        w = wordcloud.WordCloud(
            width=600,
            height=600,  # 设置图片长宽为600p
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
        topic_file = year+'_topic.png'
        w.to_file(topic_path + topic_file)
        print(year + 'saved success')



        word_vec_matrix = embedding(year,word_list2, lda_list[0])  # 词向量获取，使用未过滤停用词的语料

        '''
            获取词向量
        '''
        center_topic_index, center_labels, ap_fit_x = ap(year,word_vec_matrix, lda_list[1])
        '''
            ap聚类
        '''
        center_topic_id = 1
        dic = {}
        for id in center_topic_index:
            dic['TopicWord:' + str(center_topic_id)] = lda_list[0][id]
            dic['TopicWeight:' + str(center_topic_id)] = lda_list[1][id]
            center_topic_id = center_topic_id + 1
        center_word_df = pd.DataFrame(dic)
        final_result_path='lda_all_result/final_result/'+year+'/'
        final_result_file='final_result.csv'
        center_word_df.to_csv(final_result_path + final_result_file, sep=',')

        print("Silhouette Score: %0.5f" % metrics.silhouette_score(ap_fit_x, center_labels, metric='euclidean'))  # 轮廓系数
        print("Calinski Harabasz Score: %0.5f" % metrics.calinski_harabasz_score(ap_fit_x, center_labels))  # CH
        print("Davies Bouldin Score: %0.5f" % metrics.davies_bouldin_score(ap_fit_x, center_labels))  # DBI

lda()



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
import pyLDAvis.gensim
result_path = '2001_2020_lda10030_ap_result/'  # 输出结果及模型保存路径
lda_result_path = result_path  # lda结果存放路径
lda_result_file = '2001_2020lda_result.txt'  # lda结果保存文件名
fasttext_path = result_path  # fasttext模型路径
fasttext_file = 'fasttext.model'  # fasttext模型保存文件名
word_list_path = result_path  # 词表保存路径
word_list_file = 'wordlist2001_2020.txt'  # 词表保存文件名
tfidf_path = result_path  # tfidf保存路径
tfidf_file = 'tfidf'  # tfidf保存文件名
lda_model_path = result_path  # 保存lda模型的目录
lda_model_file = '2001_2020lda.model'  # lda模型保存文件名
lda_topic_emb_path = result_path
lda_topic_emb_file = 'lda_topic_emb.txt'  # 保存主题向量矩阵
lda_word_emb_path = result_path
lda_word_emb_file = 'lda_word_emb.txt'
final_result_path = result_path
final_result_file = '2001_2020final_result.csv'
img_result_path = result_path
img_result_file = 'img_result.png'
drop_below = 5  # 删除低频词->至少在i个条目中出现
drop_above = 0.5  # 删除高频词->在i%及以上的条目中出现
topic_count = 100  # lda模型主题数量
topn_word = 30  # 每个主题选取的词数
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

def lda(corpus, dictionary):
    try:
        lda_model = models.ldamodel.LdaModel.load(lda_model_path + lda_model_file)
        print('读取lda模型成功')
    except:
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=topic_count,
            passes=2,
            update_every=0,
            alpha='auto',
            iterations=500
        )
        '''
        topic_i -> [(word,weight),(word,weight)...]
        topic_i -> [(word,weight),(word,weight)...]
        topic_i -> [(word,weight),(word,weight)...]
        ...
        '''
        lda_model.save(lda_model_path + lda_model_file)  # 保存模型

        print('lda模型构建成功')

    lda_word_list = []
    lda_weight_list = []
    lda_final_list = []
    for i in range(topic_count):
        lda_word = []
        lda_weight = []
        for j in range(topn_word):
            lda_word.append(lda_model.show_topic(i, topic_count)[j][0])  # 获取单词矩阵
            lda_weight.append(lda_model.show_topic(i, topic_count)[j][1])  # 获取值矩阵
        lda_word_list.append(lda_word)
        lda_weight_list.append(lda_weight)
    lda_final_list = [lda_word_list,lda_weight_list]
    with open(lda_result_path+lda_result_file, 'w') as f:
        f.write(str(lda_final_list))
    # 写入格式为[[[]],[[]]]
    return lda_final_list


def ap(word_vec_matrix, weight_matrix):
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

    with open(lda_topic_emb_path + lda_topic_emb_file, 'w') as f:
        f.write(str(sum_matrix.tolist()))

    ap = AffinityPropagation(
        affinity='euclidean',  # 尝试欧氏距离
    ).fit(sum_matrix)
    return ap.cluster_centers_indices_, ap.labels_, sum_matrix  # 中心主题词的索引,标签


def embedding(word_list, word_need):
    # word_list是一个二维列表[[句子1]，[句子2].......]
    # word_need是一个二维列表[[词1，词2...]，[词1，词2...]...]
    sentences = word_list
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
    with open(lda_word_emb_path + lda_word_emb_file, 'w') as f:
        word_vec_matrix_listtosave = np.array(word_vec_matrix).tolist()
        f.write(str(word_vec_matrix_listtosave))
    print('word_vec_matrix的维度' + str(np.array(word_vec_matrix_listtosave).shape))
    return word_vec_matrix
'''
https://blog.csdn.net/qq_38890412/article/details/104710375?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167482663916800188511127%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167482663916800188511127&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-104710375-null-null.142^v71^one_line,201^v4^add_ask&utm_term=gensim.models%20FastText&spm=1018.2226.3001.4187
'''

def tsneimg(sum_matrix, label):
    tsne_result = TSNE(n_components=2).fit_transform(sum_matrix)  # （30，2）的np.array
    x = [i[0] for i in tsne_result]
    y = [i[1] for i in tsne_result]
    color = [colors[l+15] for l in label]
    plt.figure(1)
    plt.clf()
    for i in range(topic_count):
        plt.scatter(x[i], y[i], color=color[i])
    plt.savefig('tsne.png')


word_lists1=[]
word_lists2=[]
#加载分词数据 word_list2是过滤标点的  word_list1是过滤标点和停用词的
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
        word_lists1.append(word_list)
with open('words_1.csv','r',encoding='utf-8')as f:
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
#构建字典与词袋
dictionary = Dictionary(word_lists1)  # 构建gensim字典，包含词编号，词频，词名
dictionary.filter_extremes(no_below = drop_below, no_above = drop_above)  # 过滤高低频词
corpus = [dictionary.doc2bow(text) for text in word_lists1]

# 为每条数据构建词袋，二维list[[(id,doc_count),(id,doc_count)],[(id,doc_count),(id,doc_count)]]
'''
词典、词袋构建完成
'''
# 第三步：构建tfidf模型
# 一种用于资讯检索与资讯探勘的常用加权技术。TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度
# 单词的重要性随着它在文档中出现的次数而增加，随着它在所有文档中出现的频率而下降。
# 参考文献：http://www.ruanyifeng.com/blog/2013/03/tf-idf.html
try:
    tfidf = models.tfidfmodel.TfidfModel.load(tfidf_path + tfidf_file)
    print('读取tfidf模型成功')
except:
    tfidf = models.TfidfModel(corpus)
    tfidf.save(tfidf_path + tfidf_file)
    print('tfidf模型构建完成')
corpus_tfidf = tfidf[corpus]
'''
Tfidf构建完成
https://blog.csdn.net/weixin_51143561/article/details/122541859?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167480516616782425143562%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167480516616782425143562&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-122541859-null-null.142^v71^one_line,201^v4^add_ask&utm_term=models.TfidfModel&spm=1018.2226.3001.4187
'''
lda_list = lda(corpus_tfidf, dictionary)
#lda_list格式[[主题词二维列表],[对应的频度二维列表] ]
#outputpyLDAvis(dictionary,corpus)
'''
    构建lda模型
    获取一个列表：
    [[单词矩阵],[值矩阵]] -> topic_count * topn_word * 2
'''
word_vec_matrix = embedding(word_lists2, lda_list[0])  # 词向量获取，使用未过滤停用词的语料

'''
    获取词向量
'''
center_topic_index, center_labels, ap_fit_x = ap(word_vec_matrix, lda_list[1])
'''
    ap聚类
'''
tsneimg(ap_fit_x, center_labels)
'''
    TSNE绘图
'''
center_topic_id = 1
dic = {}
for i in center_topic_index:
    dic['TopicWord:' + str(center_topic_id)] = lda_list[0][i]
    dic['TopicWeight:' + str(center_topic_id)] = lda_list[1][i]
    center_topic_id = center_topic_id + 1
center_word_df = pd.DataFrame(dic)
center_word_df.to_csv(final_result_path + final_result_file, sep=',')

print("Silhouette Score: %0.5f" % metrics.silhouette_score(ap_fit_x, center_labels, metric='euclidean'))  # 轮廓系数
print("Calinski Harabasz Score: %0.5f" % metrics.calinski_harabasz_score(ap_fit_x, center_labels))  # CH
print("Davies Bouldin Score: %0.5f" % metrics.davies_bouldin_score(ap_fit_x, center_labels))  # DBI


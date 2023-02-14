import numpy as np
import pandas as pd
from gensim.models import FastText
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import  matplotlib.pyplot as plt

topic_topn_count = 30
word_topn_count = 20
topic_simvalue = 0.85
word_simvalue = 0.45
topic_alpha = 0.85
word_alpha = 0.85
topic_iter_num = 30
word_iter_num = 50
result_path = '2001_2020_lda10030_ap_result/'
lda_topic_emb = 'lda_topic_emb.txt'
lda_word_emb = 'lda_word_emb.txt'
fasttext = 'fasttext.model'
pagerank_result_path = 'pagerank_' + 'result/'
topic_pr_result = 'topic_pr_result.txt'
word_pr_result = 'word_pr_result.txt'

colors = [
    '#F0F8FF',
    '#FAEBD7',
    '#00FFFF',
    '#7FFFD4',
    '#F0FFFF',
    '#F5F5DC',
    '#FFE4C4',
    '#000000',
    '#FFEBCD',
    '#0000FF',
    '#8A2BE2',
    '#A52A2A',
    '#DEB887',
    '#5F9EA0',
    '#7FFF00',
    '#D2691E',
    '#FF7F50',
    '#6495ED',
    '#FFF8DC',
    '#DC143C',
    '#00FFFF',
    '#00008B',
    '#008B8B',
    '#B8860B',
    '#A9A9A9',
    '#006400',
    '#BDB76B',
    '#8B008B',
    '#556B2F',
    '#FF8C00',
    '#9932CC',
    '#8B0000',
    '#E9967A',
    '#8FBC8F',
    '#483D8B',
    '#2F4F4F',
    '#00CED1',
    '#9400D3',
    '#FF1493',
    '#00BFFF',
    '#696969',
    '#1E90FF',
    '#B22222',
    '#FFFAF0',
    '#228B22',
    '#FF00FF',
    '#DCDCDC',
    '#F8F8FF',
    '#FFD700',
    '#DAA520',
    '#808080',
    '#008000',
    '#ADFF2F',
    '#F0FFF0',
    '#FF69B4',
    '#CD5C5C',
    '#4B0082',
    '#FFFFF0',
    '#F0E68C',
    '#E6E6FA',
    '#FFF0F5',
    '#7CFC00',
    '#FFFACD',
    '#ADD8E6',
    '#F08080',
    '#E0FFFF',
    '#FAFAD2',
    '#90EE90',
    '#D3D3D3',
    '#FFB6C1',
    '#FFA07A',
    '#20B2AA',
    '#87CEFA',
    '#778899',
    '#B0C4DE',
    '#FFFFE0',
    '#00FF00',
    '#32CD32',
    '#FAF0E6',
    '#FF00FF',
    '#800000',
    '#66CDAA',
    '#0000CD',
    '#BA55D3',
    '#9370DB',
    '#3CB371',
    '#7B68EE',
    '#00FA9A',
    '#48D1CC',
    '#C71585',
    '#191970',
    '#F5FFFA',
    '#FFE4E1',
    '#FFE4B5',
    '#FFDEAD',
    '#000080',
    '#FDF5E6',
    '#808000',
    '#6B8E23',
    '#FFA500',
    '#FF4500',
    '#DA70D6',
    '#EEE8AA',
    '#98FB98',
    '#AFEEEE',
    '#DB7093',
    '#FFEFD5',
    '#FFDAB9',
    '#CD853F',
    '#FFC0CB',
    '#DDA0DD',
    '#B0E0E6',
    '#800080',
    '#FF0000',
    '#BC8F8F',
    '#4169E1',
    '#8B4513',
    '#FA8072',
    '#FAA460',
    '#2E8B57',
    '#FFF5EE',
    '#A0522D',
    '#C0C0C0',
    '#87CEEB',
    '#6A5ACD',
    '#708090',
    '#FFFAFA',
    '#00FF7F',
    '#4682B4',
    '#D2B48C',
    '#008080',
    '#D8BFD8',
    '#FF6347',
    '#40E0D0',
    '#EE82EE',
    '#F5DEB3',
    '#FFFFFF',
    '#F5F5F5',
    '#FFFF00',
    '#9ACD32'
]

def read_file(topic_emb_path, word_emb_path):
    with open(topic_emb_path, 'r') as f:
        lda_topic_emb = eval(f.read())
    with open(word_emb_path, 'r') as f:
        lda_word_emb = eval(f.read())
        print('lda_word_emb的维度：' + str(np.array(lda_word_emb).shape))
    return lda_topic_emb, lda_word_emb
    """
    读取文件：
    1)topicCount * dim 主题向量矩阵
    2)topicCount * wordCount * dim 主题词向量矩阵
    """


def get_dis(emb_matrix):  # 输入应为主题数（单词数）*维度的矩阵，这里使用数组
    sim_vec = 1 - pdist(emb_matrix, 'cosine')  # 获取相似性矩阵，为简洁矩阵
    sim_matrix = squareform(sim_vec)
    # print(sim_matrix)
    return sim_matrix


def weight_pagerank(sim_value_matrix, simvalue, alpha, iter_num, topn):  # 输入为二维列表（相似度矩阵）
    sim_value_matrix_np = np.array(sim_value_matrix)  # 输入的列表转化为numpy数组
    count = sim_value_matrix_np.shape[0]  # 获取原始topic数量
    print('相似度矩阵原始维度：' + str(count) + '*' + str(count))
    condtion = sim_value_matrix_np > simvalue  # 筛选相似度大于阈值的边
    arg_all0 = np.zeros(count)  # 用于保存全0的行（列），全0则为0，否则为1
    sim_value_matrix_np_0 = np.zeros((count, count))  # 创建数组接收归零的结果
    for i in range(count):
        for j in range(count):
            if condtion[i][j]:
                arg_all0[i] = 1  # 表示此行非全0
            sim_value_matrix_np_0[i][j] = sim_value_matrix_np[i][j] if condtion[i][j] else 0

    '''
    用condition统计每行非0值的个数，用于normaliza
    '''
    edge_count = []
    for i in range(count):
        num = 0
        for j in range(count):
            if condtion[i][j]:
                num = num + 1
        edge_count.append(num)

    '''
    normaliza
    '''
    for i in range(count):
        for j in range(count):
            if edge_count[i]:
                sim_value_matrix_np_0[i][j] = sim_value_matrix_np_0[i][j] / edge_count[i]

    '''
    删除边权重全为0的点'''

    # for target, i in zip(arg_all0, range(count)):
    #     if not target:
    all0_arg = np.argwhere( arg_all0 == 0).reshape(-1).tolist()  # 获取全0行的索引列表
    sim_value_matrix_np_0 = np.delete(sim_value_matrix_np_0, all0_arg, axis=0)
    # print('删除全0行之后的维度：' + str(sim_value_matrix_np_0.shape))
    # for target, i in zip(arg_all0, range(count)):
    #     if not target:
    sim_value_matrix_np_0 = np.delete(sim_value_matrix_np_0, all0_arg, axis=1)
    # print('删除全0列之后的维度：' + str(sim_value_matrix_np_0.shape))
    sim_value_matrix_np_del0 = sim_value_matrix_np_0
    count_new = sim_value_matrix_np_del0.shape[0]
    print('去除无边node之后的维度:(选取的topN的PR值应小于等于这个值）' + str(sim_value_matrix_np_del0.shape))

    vec = np.array([1/count_new for _ in range(count_new)])  # 构建初始PR值向量，格式为numpy数组
    print('初始向量构建完成')

    vec_history = []  # 保存vec的变化历史，用来画图
    for iter_time in range(iter_num):
        vec = (1 - alpha) + alpha * np.dot(sim_value_matrix_np_del0, vec)
        vec_history.append(vec)  # iter_num * count_new

    vec_sort = np.argsort(vec)  # 获取PR值的索引排序

    org_arg = []  # 记录在原始topic列表中的索引
    for i in range(count):
        if arg_all0[i]:
            org_arg.append(i)
    for i in range(count_new):
        vec_sort[i] = org_arg[vec_sort[i]]
        # 将索引排序结果中的索引还原为在原始topic列表中的索引

    # print('vec_history的维度:' + str(np.array(vec_history).shape))
    vec_history = np.array(vec_history).T

    return vec_history, vec_sort[-topn:]


if __name__ == '__main__':
    topic_emb_path = result_path + lda_topic_emb
    word_emb_path = result_path + lda_word_emb
    topic_emb, word_emb = read_file(topic_emb_path, word_emb_path)

    topic_dis_matrix = get_dis(topic_emb)
    topic_pr_history, topn_topic_arg = weight_pagerank(
        topic_dis_matrix,
        topic_simvalue,
        topic_alpha,
        topic_iter_num,
        topic_topn_count)  # 列表，顺序为从小到大！
    topn_word_arg_all = []
    print('word_emb的维度：' + str(np.array(word_emb).shape))

    x = [iter_time for iter_time in range(topic_iter_num)]
    # print('np.array(topic_pr_history).shape[0]:' + str(np.array(topic_pr_history).shape[0]))
    # plt.xlim(0,2)
    # plt.ylim(0.1,0.2)
    for i in range(np.array(topic_pr_history).shape[0]):
        plt.plot(x, topic_pr_history[i], color=colors[i])
    plt.savefig(pagerank_result_path + 'topicPrValue.png')

    word_pr_history_all = []
    j=0
    for topic in topn_topic_arg:
        if(j<16):
            j=j+1
            continue
        word_dis_matrix = get_dis(np.array(word_emb[topic]))

        word_pr_history,topn_word_arg = weight_pagerank(
            word_dis_matrix,
            word_simvalue,
            word_alpha,
            word_iter_num,
            word_topn_count)
        topn_word_arg_all.append(topn_word_arg)
        word_pr_history_all.append(word_pr_history)  # 应为topNtopic*topNword*iter_num

    # x = [iter_time for iter_time in range(word_iter_num)]
    # for word_pr_history, i in zip(word_pr_history_all, range(i)):
    #
    #     for

    print('Top' + str(topic_topn_count) + '-topic-arg:')
    print(topn_topic_arg)
    with open(pagerank_result_path + topic_pr_result, 'w') as f:
        f.write(str(topn_topic_arg))

    print('Top' + str(word_topn_count) + '-word-arg:')
    print(np.array(topn_word_arg_all).tolist())
    with open(pagerank_result_path + word_pr_result, 'w') as f:
        f.write(str(np.array(topn_word_arg_all).tolist()))




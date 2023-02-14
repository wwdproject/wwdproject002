import numpy as np
import pandas as pd

lda_result_path = '2001_2020_lda10030_ap_result/'
lda_result_file = '2001_2020lda_result.txt'
pagerank_result_path = 'pagerank_result/'
topic_pr_result_file = 'topic_pr_result.txt'
word_pr_result_file = 'word_pr_result.txt'
new_word_list_file = 'pr_topic_word.csv'
'''
     

     '''

def read_file():
    with open(pagerank_result_path + topic_pr_result_file, 'r') as f:
        txt_list = f.read()
        txt_list = txt_list.strip('[').strip(']')
        txt_list = txt_list.split(' ')
        topic_result = [word for word in txt_list if word is not None]
    #    topic_result = eval(f.read())
    with open(pagerank_result_path + word_pr_result_file, 'r') as f:
        word_result = eval(f.read())
    with open(lda_result_path + lda_result_file, 'r') as f:
        lda_result = eval(f.read())
        print(len(lda_result[0][0]))

    return topic_result, word_result, lda_result[0]
# 返回一维列表,二维列表,三维列表


def getword(topic_arg, word_arg, word_list):
    new_word_list_all = []
    for topic, word in zip(topic_arg, word_arg):
        new_word_list = []
        try:
            topic=int(topic)
        except:
            continue
        print(np.array(word_arg).tolist(), topic)
        print(len(word_list[1]))
        for word_index in word:
            new_word_list.append(word_list[int(topic)][word_index])
        new_word_list_all.append(new_word_list)

    return new_word_list_all
# 返回新单词表，按权重从大到小排序


def saveword(word_list):
    dic = {}
    rank = 0
    for topic in word_list:
        rank = rank + 1
        dic['Topic' + str(rank)] = topic
    top_topic = pd.DataFrame(dic)
    top_topic.to_csv(new_word_list_file, sep=',')
    print('词表保存成功')


if __name__ == '__main__':
    topic_arg, word_arg, word_list = read_file()
    word_list = getword(topic_arg, word_arg, word_list)
    saveword(word_list)
import jieba
import xlrd
import re
import string
from zhon.hanzi import punctuation
data_file="data.xls"
word_file="word_file.xls"
word_path='word_file/'
word_type1='word_savedbyyear/'  #根据年份存储
word_type2='word_savedbyarea/'  #根据学部存储

ids=['1','2','3','4','5','6','7','8','9','U','J','L']
labels=['数理科学部','化学科学部','生命科学部','地球科学部','工程与材料科学部','信息科学部','管理科学部','医学科学部','交叉科学部','联合基金','专项项目','应急管理项目']
id_num=[0,1,2,3,4,5,6,7,8,9,10,11]
id_labels=dict(zip(ids,labels))
label_nums=dict(zip(labels,id_num))
#加载中文停用词
dropwords=[]
with open('停用词','r',encoding='utf-8') as f:
    words=f.readlines()
    for word in words:
        word=word.strip('\n')
        dropwords.append(word)
#加载中文标点
punctions=punctuation
punctions=punctions+string.punctuation
chinese_punctions=[]
for i in range(len(punctions)):
    chinese_punctions.append(punctions[i])


def analyseid(id):
    #根据项目号返回所属学部，立项时间
    first_word=id[0:1]
    years_word='20'+id[1:3]
    return id_labels[first_word],years_word
def savefile(project_items):
    #首先按年份保存
    i=0
    for item in project_items:
        with open(word_path+word_type1+item[0][1]+'/'+item[0][0]+'_1.csv','a',encoding='utf-8') as f1:
            with open(word_path+word_type1+item[0][1]+'/'+item[0][0]+'_2.csv','a',encoding='utf-8') as f2:
                f1.write(str(item[1]))
                f1.write('\n')
                f2.write(str(item[2]))
                f2.write('\n')
                print(i)
                i=i+1

    print('words_lists has been saved by years')
    #按学部保存
    i=0
    for item in project_items:
        with open(word_path+word_type2+item[0][0]+'/'+item[0][1]+'_1.csv','a',encoding='utf-8') as f1:
            with open(word_path+word_type2+item[0][0]+'/'+item[0][1]+'_2.csv','a',encoding='utf-8') as f2:
                f1.write(str(item[1]))
                f1.write('\n')
                f2.write(str(item[2]))
                f2.write('\n')
                print(i)
                i=i+1

    print('words_lists has been saved by areas')

def createdicbyarea(txts): #根据所属学部构造字典
    dic_area=[]
    for i in range(12):
        area=[]
        dic_area.append(area)
    for txt in txts:
        label_id=label_nums[txt[0]]
        dic_area[label_id]=dic_area[label_id]+txt[1]
    for i in range(12):
        dic_area[i]=list(set(dic_area[i]))
        print(i)
        with open('word/'+labels[i]+'_topic.csv','w',encoding='utf-8')as f:
            for txt in dic_area[i]:
                if(txt is None or txt==' ' or txt=='\n'):
                    continue
                f.write(txt)
                f.write('\n')
    print('save success')


def pretreatment(txts,area_name):
    #使用jieba工具获取分词
    jieba.load_userdict('word/'+area_name+'_topic.csv')
    jieba.load_userdict('luser_dic2.txt')
    txt_list= jieba.cut(txts, cut_all=False)
    txt_list=list(txt_list)
    print('jieba 中')
    #去除标点
    #filtered_words = [[word for word in text if word not in chinese_punctions] for text in txts]  # 过滤标点
    filtered_words1=[word for word in txt_list if word not in chinese_punctions and word != ' ' and word !='篇']
    #去除停用词
    filtered_words2=[word for word in filtered_words1 if word not in dropwords and word != ' 'and word !='篇']
    return filtered_words1,filtered_words2
# 打开实验数据表格
book = xlrd.open_workbook(data_file)
# 选择页数为第1页
sheet1 = book.sheets()[0]
# 数据总行数
nrows = sheet1.nrows
project_items=[]
#构建数据列表，每个项目列表包含信息：项目号，项目所属学部，项目年份,项目类别，项目中文简介
#本阶段保留项目中文简介,并根据年份，所属学部将之分别保存
txts=[]
for i in range(nrows):
    if(i==0): #不需要第一行数据
        continue
    txt_list=sheet1.row_values(i)
    #格式['项目批准号', '批准后项目中文名称', '资助类别', '批准金额（万元）', '项目中文主题词', '中文摘要']
    project_item=[]
    project_title0=txt_list[4].split('；') #此为项目自称主题
    title=[]
    for words in project_title0:   #有的项目用英文符号
        words=str(words).split(';')
        title=title+words
    title2=[]
    for words in title:
        words=str(words).split('，')
        title2=title2+words
    project_txt=txt_list[5]               #此为项目中文摘要
    if(project_txt is None or project_txt==' '):
        continue
    project_id=txt_list[0]
    id_label=analyseid(str(project_id))


    words1,words2=pretreatment(project_txt,id_label[0])
    project_item.append(id_label)
    project_item.append(words1)
    project_item.append(words2)
    project_items.append(project_item)

    '''
    txt=[]
    txt.append(id_label[0])
    txt.append(title2)
    txts.append(txt)
    '''

#createdicbyarea(txts)


savefile(project_items)

'''
with open('words_1.csv', 'w',encoding='utf-8') as f1:
    with open('words_2.csv','w',encoding='utf-8') as f2: #保存分词结果
        for item in project_items:
            f1.write(str(item[0]))
            f1.write('\n')
            f2.write(str(item[1]))
            f2.write('\n')
print('words_lists has been saved')
print('文本预处理完成')
'''








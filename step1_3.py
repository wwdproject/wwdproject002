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

def analyseid(id):
    #根据项目号返回所属学部，立项时间
    first_word=id[0:1]
    years_word='20'+id[1:3]
    return id_labels[first_word],years_word

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
    '''
      project_money=txt_list[3]
    project_money=str(project_money)
    money_path='money_file/'+id_label+'/'
    
    '''
    project_title0 = txt_list[4].split('；')  # 此为项目自称主题
    title = []
    for words in project_title0:  # 有的项目用英文符号
        words = str(words).split(';')
        title = title + words
    title2 = []
    for words in title:
        words = str(words).split('，')
        title2 = title2 + words
    project_id=txt_list[0]
    id_label,year=analyseid(str(project_id))
    topic_path='topic_file/'+id_label+'/'
    topic_file=year+'_topic.txt'
    with open(topic_path+topic_file,'a') as f:
        for title_item in title2:
            f.write(title_item)
            f.write('\n')
    print(i)
print('save success')


'''
    txt=[]
    txt.append(id_label[0])
    txt.append(title2)
    txts.append(txt)
'''

#createdicbyarea(txts)










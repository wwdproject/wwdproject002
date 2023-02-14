import jieba
import xlrd
import re
import string
from zhon.hanzi import punctuation
data_file="data.xls"
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
letters=string.ascii_lowercase+string.ascii_uppercase+'-'+'_'+'0123456789'
letters2=letters+' '
#包含a-z的小写字母
def findeng(project_txt):
    #找到中文摘要里英文专有名词
    lenth=len(project_txt)
    word_engs=[]
    i=0
    while(i<lenth):
        if(project_txt[i] in letters):
            word_english=''
            while( i<lenth and project_txt[i] in letters2):
                word_english+=project_txt[i]
                i=i+1
            word_english=word_english.strip()
            if(len(word_english)!=1):
                word_engs.append(word_english)
        i=i+1
    return list(set(word_engs))



word_all=[]
for i in range(nrows):
    if(i==0): #不需要第一行数据
        continue
    txt_list=sheet1.row_values(i)


    project_txt=txt_list[5]               #此为项目中文摘要
    if(project_txt is None or project_txt==' '):
        continue
    word_all=word_all+findeng(project_txt)
    word_all=list(set(word_all))
with open('luser_dic2.txt','w')as f:
    for word in word_all:
        f.write(word)
        f.write('\n')



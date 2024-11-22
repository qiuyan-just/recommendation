from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import pkg_resources
import nltk
from nltk.tokenize import word_tokenize
from language_detector import detect_language
import jieba
import re

'sentence level preprocessing'
def f_base(s):
    # 删除句子中的特殊符号
    s = re.sub(r'[，。\{；]','',s)
    # 全部英文字母改为小写
    s = s.lower()
    # 删除文本中可能包含的大量空格和空白行
    s =s.strip()
    return s


def preprocess_sent(rw):
    # 返回sentence level pre-processed
    s = f_base(rw)
    return s



'word level preprocessing'
# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('./data/cn_stopwords.txt', 'r', encoding='utf-8').readlines()]
    return stopwords

# 对句子进行中文分词
def preprocess_word(w_list):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(w_list)
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = []
    # 去停用词 这是因为在好大夫数据集上需要去除停用词，这些词语是无用的，不包含实际含义
    for word in sentence_depart:
        if word not in stopwords and word !="\t" and len(word) != 1:  ###设置len(word) != 1是好大夫数据集上使用
            outstr.append(word)
            # outstr += word
            # outstr += " "
    return outstr

    # ## 使用百度数据集，为了验证有效性，不需要取出停用词，因为有时候提问的问题可能都是些停用词构成的
    # for word in sentence_depart:
    #     outstr.append(word)
    # return outstr






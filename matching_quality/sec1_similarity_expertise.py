
# from model import *
import numpy as np
from model_qiuyan1 import *
from model_qiuyan2 import *
from utils import *
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt
import itertools
import argparse

import warnings
warnings.filterwarnings('ignore', category=Warning)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'


### 此处选择方法，包括 TFIDF、LDA、MPNet、LDA_MPNet
method = "LDA_MPNet"
samp_size = 50000
ntopic = 10



'''
计算新问题与医生之前回答中体现的expertise相似性计算
'''

## 定义相似度函数
def KL(p, q):
    # p,q 为两个 list，表示对应取值的概率 且 sum(p) == 1 ，sum(q) == 1
    return sum(_p * math.log(_p / _q) for (_p, _q) in zip(p, q) if _p != 0)

def JS(p, q):
    M = [0.5 * (_p + _q) for (_p, _q) in zip(p, q)]
    return 0.5 * (KL(p, M) + KL(q, M))


#一维向量和多维向量之间的相似度函数定义
def get_cos_similar_multi(v1, v2):
    num = np.dot([v1], np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

#多维向量之间的相似度函数定义
def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


# -------------------------------使用训练好的模型计算两个文档的相似度-----------------------------------
def lda_bert_sim(data1, data2):

    '''
    第一篇文档
    :param data1:
    :param data2:
    :return:
    '''
    data1 = data1.fillna('')  # only the comments has NaN's
    rws1 = data1.que
    # sentences1, token_lists1, idx_in1 = preprocess(rws1, samp_size=int(args.samp_size))
    sentences1, token_lists1, idx_in1 = preprocess(rws1, samp_size=samp_size)
    print(token_lists1)

    # Define the topic model object
    tm1 = Topic_Model1(k = ntopic, method = method)

    # 获取向量
    vec_ldabert1 = tm1.vectorize(sentences1, token_lists1)
    print(type(vec_ldabert1))
    # print(vec_ldabert1.ndim)
    vec1 = vec_ldabert1.tolist()
    print(len(vec1))

    #使用itertools将多维list转为一维
    # vec1 = list(itertools.chain.from_iterable(vec1))
    # print(vec1)


    '''
    第二篇文档
    '''
    data2 = data2.fillna('')  # only the comments has NaN's
    # rws2 = data2.dialogue
    rws2 = data2.que #为了使用同一个数据集提取特征
    sentences2, token_lists2, idx_in2 = preprocess(rws2, samp_size=samp_size)
    print(token_lists2)
    # sentences, token_lists= preprocess(rws, samp_size=int(args.samp_size))
    # Define the topic model object
    tm2 = Topic_Model2(k = ntopic, method = method)
    # 调用类中函数vectorize，获得连接向量返回值
    vec_ldabert2 = tm2.vectorize(sentences2, token_lists2)
    # print(vec2)
    # print(vec_ldabert1.ndim)
    vec2 = vec_ldabert2.tolist()
    print(len(vec2))

    # # 使用itertools将多维list转为一维
    # vec2 = list(itertools.chain.from_iterable(vec2))


    #计算相似性
    # # sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # sim = 1 - JS(vec1, vec2)

    # sim = get_cos_similar_multi(vec1, vec2)

    sim = get_cos_similar_matrix(vec1, vec2)

    return sim


# 输入两篇文档计算相似度
# data1 = pd.read_csv(r'./data/final_liver_time.csv', encoding='utf-8')  ## 多个问题作为第一篇文档的输入
# data2 = pd.read_csv(r'./data/liver_list_answers5.csv', encoding='gbk')  ## 多个候选医生的dialogues作为第二篇文档的输入

# 使用好大夫上的数据提取2020年的医生文本特征：LDA、MPNet、LDA+MPNet
data1 = pd.read_csv(r'./data/2020data_features.csv', encoding='utf-8')
data2 = pd.read_csv(r'./data/2020data_features.csv', encoding='utf-8')


result = lda_bert_sim(data1, data2)
print(result)
# result2 = round(result, 4)
result3 = pd.DataFrame(result) #有几个医生就定义几列
# result3.to_csv('./output/matching_quality_36-716.csv')
result3.to_csv('./output/1similarity.csv')

# # print('相似性计算结果：')
# # print(result2)









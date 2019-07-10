# -*- coding: utf-8 --*--
# @Author: Zessay
# @time: 2019.05.05 21:12
# @File: get_features.py
# @Software: PyCharm

import pickle
import jieba
import jieba.posseg as pseg
import jieba.analyse
import numpy as np
import pandas as pd
import json
import os
import re
import codecs
import logging
import multiprocessing as mp 


from gensim.models import Word2Vec, Doc2Vec, KeyedVectors
from tqdm import tqdm

import gc
import math

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.stats as stats

import warnings
warnings.filterwarnings('ignore')

from utils import loadData
from .generate_all_tokens import loadStopWords, loadDict
from config import DefaultConfig
from .train_models import *

def load_models(opt, df_):
    # 获取模型的位置
    tfidf_path = os.path.join(opt['model_word'], 'Tfidf.pkl')
    w2v_path = os.path.join(opt['model_word'], 'word2vec.model')
    d2v_path = os.path.join(opt['model_word'], 'doc2vec.model')
    # 获取TFIDF模型
    try:
        f = open(tfidf_path, 'rb')
    except:
        train_tfidf(df_)
        f = open(tfidf_path)
    finally:
        tfidf_model = pickle.load(f)
        f.close()


    # 加载word2vec和doc2vec模型
    try:
        w2v_model = Word2Vec.load(w2v_path)
    except:
        train_word2vec(df_)
        w2v_model = Word2Vec.load(w2v_path)

    try:
        d2v_model = Doc2Vec.load(d2v_path)
    except:
        train_doc2vec(df_)
        d2v_model = Doc2Vec.load(d2v_path)

    return tfidf_model, w2v_model, d2v_model

# 计算单词与词向量的余弦相似度和欧氏距离
## 计算余弦相似度
def Cosine(wordvec, docvec):
    wordvec, docvec = np.array(wordvec), np.array(docvec)
    return wordvec.dot(docvec) / (math.sqrt((wordvec**2).sum()) * math.sqrt((docvec**2).sum()))

## 计算欧式距离
def Euclidean(wordvec, docvec):
    wordvec, docvec = np.array(wordvec), np.array(docvec)
    return math.sqrt(((wordvec-docvec)**2).sum())

# 判断是否是数字
def is_number(x):
    try:
        float(x)
        return True
    except:
        return False


def extract_word_property(news, tfidf_model, w2v_model, d2v_model, train=True, train_news=None):
    tmp = pd.DataFrame()
    text = news['title'] + '\n' + news['content']
    # words = pseg.cut(text)
    # print(list(zip(*words)))
    # li = list(zip(*words))
    # print(li[0])
    assert len(news['tokens'].split(',')) == len(news['cixing'].split(',')), '长度不匹配'

    tmp['word'] = news['tokens'].split(',')
    tmp['word'] = tmp['word'].apply(lambda w: w.strip())
    tmp['cixing'] = news['cixing'].split(',')

    tmp['newsId'] = news['newsId']
    tmp['lda_classes'] = news['lda_classes']
    tmp['kmeans_classes'] = news['kmeans_classes']
    # 添加词频的列
    tmp.dropna(axis=0, subset=['word'], inplace=True)
    tmp['tf'] = tmp['word'].apply(lambda w: text.count(w))
    tmp.drop_duplicates(inplace=True)


    stopwords = loadStopWords(stop_path)
    # 添加是否为停用词的标记，或者为空，或者为纯数字
    tmp['flag'] = tmp['word'].apply(lambda w: int((w in stopwords) or (is_number(w))))
    # 删除停用词，取出非停用词
    tmp = tmp[tmp['flag'] == 0]
    tmp.drop(['flag'], axis=1, inplace=True)

    # 增加词的长度特征
    tmp['word_len'] = tmp['word'].apply(lambda w: len(w))
    ## 相对于最大词长的比值
    max_len = max(tmp['word_len'].values)
    tmp['word_len_ratio'] = tmp['word_len'].apply(lambda x: x / max_len)
    # 去除词长小于1的
    tmp = tmp[tmp.word_len > 1]

    # 计算频率
    tmp['tf_ratio'] = tmp['tf'] / sum(tmp['tf'])

    ## 获取所有的实体
    feature_name = tfidf_model.get_feature_names()
    # 获取TfIdf
    tfidf_fetures = tfidf_model.transform([news['tokens']])
    tfidf_map = {feature_name[k]: v for k, v in zip(tfidf_fetures.indices, tfidf_fetures.data)}
    tmp['tfidf'] = tmp['word'].map(tfidf_map)
    ## 对TFIDF进行归一化
    tmp['tfidf'] = normalize(tmp['tfidf'].fillna(0).values.reshape(1, -1), 'max')[0]


    # 获取textrank
    tr_score = jieba.analyse.textrank(text, topK=200, withWeight=True, allowPOS=list(tmp['cixing'].unique()))
    tr_map = {k: v for k, v in tr_score}
    tmp['text_rank'] = tmp['word'].map(tr_map)

    # 返回前100
    #tmp = tmp if tmp.shape[0] <= 100 else tmp.iloc[:100, :]


    # 找出第一次出现的位置
    tmp['first_ocur'] = tmp['word'].apply(lambda w: text.find(w) + len(w) - 1)
    # 找出最后一次出现的位置
    reverse_text = text[-1::-1]
    all_len = len(text)
    tmp['last_ocur'] = tmp['word'].apply(lambda w: all_len - (reverse_text.find(w[-1::-1]) + len(w)))
    # 计算词跨度
    tmp['word_distance'] = (tmp['last_ocur'] - tmp['first_ocur']).apply(lambda d: 0 if d < 0 else d)
    # 对词跨度进行归一化
    tmp['word_distance_norm'] = normalize(tmp['word_distance'].values.reshape(1, -1), 'max')[0]
    tmp.drop(['first_ocur', 'last_ocur', 'word_distance'], axis=1, inplace=True)

    # 计算词向量和文档向量的余弦距离
    tmp['Cosine'] = tmp['word'].apply(
        lambda w: np.nan if w not in w2v_model.wv.vocab else Cosine(w2v_model.wv[w], d2v_model[news['newsId']]))

    # 计算词向量和文档向量的欧式距离
    tmp['Euclidean'] = tmp['word'].apply(
        lambda w: np.nan if w not in w2v_model.wv.vocab else Euclidean(w2v_model.wv[w], d2v_model[news['newsId']]))

    # 计算皮尔逊相关系数
    tmp['pearson_cor'] = tmp['word'].apply(
        lambda w: np.nan if w not in w2v_model.wv.vocab else stats.pearsonr(w2v_model.wv[w], d2v_model[news['newsId']])[
            0])
    tmp['pearson_pvalue'] = tmp['word'].apply(
        lambda w: np.nan if w not in w2v_model.wv.vocab else stats.pearsonr(w2v_model.wv[w], d2v_model[news['newsId']])[
            1])

    # 是否出现在标题中
    tmp['ocur_in_title'] = tmp['word'].apply(lambda w: int((1 if news['title'].find(w) != -1 else 0)))

    # 是否还有数字，是否含有字母
    tmp['has_num'] = tmp['word'].apply(lambda w: int(bool(re.search(r'\d', w))))
    tmp['has_char'] = tmp['word'].apply(lambda w: int(bool(re.search(r'[a-zA-Z]+', w))))

    # 添加共现矩阵信息
    sentences = [news['title']]
    for seq in re.split(r'[\n。？！?!.]', news['content']):
        # 如果开头不是汉字、字母或数字，则去除
        seq = re.sub(r'^[^\u4e00-\u9fa5A-Za-z0-9]+', '', seq)
        # 去除之后，如果句子不为空，则添加进句子中
        if len(seq) > 0:
            sentences.append(seq)

    num_tokens = len(tmp['word'])
    words_list = tmp['word'].tolist()
    arr = np.zeros((num_tokens, num_tokens))
    # 得到共现矩阵
    for i in range(num_tokens):
        for j in range(i + 1, num_tokens):
            count = 0
            for sentence in sentences:
                if (words_list[i] in sentence) and (words_list[j] in sentence):
                    count += 1
            arr[i, j] = count
            arr[j, i] = count

    ## 得到偏度的统计特征
    tmp['coocur_skew'] = stats.skew(arr)
    ## 计算某个单词共现矩阵的方差、峰度
    tmp['coocur_var'] = np.var(arr, axis=0)
    tmp['coocur_mean'] = np.mean(arr, axis=0)
    tmp['coocur_kurt'] = stats.kurtosis(arr, axis=0)
    tmp['coocur_std'] = np.std(arr, axis=0)
    #tmp['coocur_median'] = np.median(arr, axis=0)
    #tmp['coocur_variation'] = stats.variation(arr, axis=0)

    ## 共现矩阵一阶差分
    co_diff1 = np.diff(arr, n=1, axis=1)
    tmp['diff_coocur_mean'] = np.mean(co_diff1, axis=1)
    tmp['diff_coocur_var'] = np.var(co_diff1, axis=1)
    tmp['diff_coocur_std'] = np.std(co_diff1, axis=1)
    #tmp['diff_coocur_median'] = np.median(co_diff1, axis=1)
    tmp['diff_coocur_skew'] = stats.skew(co_diff1, axis=1)
    tmp['diff_coocur_kurt'] = stats.kurtosis(co_diff1, axis=1)

    '''
    ## 共现矩阵二阶差分的统计特性
    co_diff2 = np.diff(arr, n=2, axis=1)
    tmp['diff2_coocur_mean'] = np.mean(co_diff2, axis=1)
    tmp['diff2_coocur_var'] = np.var(co_diff2, axis=1)
    tmp['diff2_coocur_std'] = np.std(co_diff2, axis=1)
    tmp['diff2_coocur_median'] = np.median(co_diff2, axis=1)
    tmp['diff2_coocur_skew'] = stats.skew(co_diff2, axis=1)
    tmp['diff2_coocur_kurt'] = stats.kurtosis(co_diff2, axis=1)
    '''

    # 计算相似度矩阵以及统计信息
    ## sim_tags_arr: 初始化候选词相似度矩阵
    sim_tags_arr = np.zeros((num_tokens, num_tokens))

    for i in range(num_tokens):
        for j in range(i, num_tokens):
            sim_tags_arr[i][j] = 0 if (words_list[i] not in w2v_model.wv.vocab or words_list[
                j] not in w2v_model.wv.vocab) else w2v_model.wv.similarity(words_list[i], words_list[j])
            if i != j:
                sim_tags_arr[j][i] = sim_tags_arr[i][j]
    # 计算单词相似度矩阵的统计信息
    ## 相似度平均值
    tmp['mean_sim_tags'] = np.mean(sim_tags_arr, axis=1)
    ## 相似度矩阵的偏度
    tmp['skew_sim_tags'] = stats.skew(sim_tags_arr, axis=1)
    ## 相似度矩阵的峰值
    tmp['kurt_sim_tags'] = stats.kurtosis(sim_tags_arr, axis=1)
    tmp['var_sim_tags'] = np.var(sim_tags_arr, axis=1)
    tmp['std_sim_tags'] = np.std(sim_tags_arr, axis=1, ddof=1)
    #tmp['median_sim_tags'] = np.median(sim_tags_arr, axis=1)
    #tmp['variation_sim_tags'] = stats.variation(sim_tags_arr, axis=1)

    ## 一阶差分统计特征补充
    diff1 = np.diff(sim_tags_arr, n=1, axis=1)
    tmp['diff_median_sim_tags'] = np.median(diff1, axis=1)
    tmp['diff_var_sim_tags'] = np.var(diff1, axis=1)
    tmp['diff_std_sim_tags'] = np.std(diff1, axis=1)
    ## 相似度矩阵的差分均值
    tmp['diff_mean_sim_tags'] = np.mean(diff1, axis=1)
    ## 相似度矩阵差分的偏度
    tmp['diff_skew_sim_tags'] = stats.skew(diff1, axis=1)
    ## 相似度矩阵差分的峰度
    tmp['diff_kurt_sim_tags'] = stats.kurtosis(diff1, axis=1)

    '''
    ## 二阶差分统计特征
    diff2 = np.diff(sim_tags_arr, n=2, axis=1)
    tmp['diff2_mean_sim_tags'] = np.mean(diff2, axis=1)
    tmp['diff2_median_sim_tags'] = np.median(diff2, axis=1)
    tmp['diff2_var_sim_tags'] = np.var(diff2, axis=1)
    tmp['diff2_std_sim_tags'] = np.std(diff2, axis=1)
    tmp['diff2_skew_sim_tags'] = stats.skew(diff2, axis=1)
    tmp['diff2_kurt_sim_tags'] = stats.kurtosis(diff2, axis=1)
    '''

    # 包含关系矩阵
    in_arr = np.zeros((num_tokens, num_tokens))
    ## 计算包含关系矩阵
    for i in range(num_tokens):
        for j in range(num_tokens):
            if i != j:
                if words_list[i] in words_list[j]:
                    in_arr[i][j] = 1

    ### 被多少个词包含
    tmp['be_include_sum'] = np.sum(in_arr, axis=1)
    tmp['be_include_mean'] = np.mean(in_arr, axis=1)
    tmp['be_include_var'] = np.var(in_arr, axis=1)
    tmp['be_include_std'] = np.std(in_arr, axis=1)
    tmp['be_include_skew'] = stats.skew(in_arr, axis=1)
    tmp['be_include_kurt'] = stats.kurtosis(in_arr, axis=1)

    ### 包含了多少个词
    tmp['include_sum'] = np.sum(in_arr, axis=0)
    tmp['include_mean'] = np.mean(in_arr, axis=0)
    tmp['include_var'] = np.var(in_arr, axis=0)
    tmp['include_std'] = np.std(in_arr, axis=0)
    tmp['include_skew'] = stats.skew(in_arr, axis=0)
    tmp['include_kurt'] = stats.kurtosis(in_arr, axis=0)

    # 对tfidf排名
    tmp = tmp.sort_values(by='tfidf', ascending=False)
    tmp['tfidf_index'] = range(tmp.shape[0])
    # 对TFIDF索引进行归一化
    tmp['tfidf_index'] = normalize(tmp['tfidf_index'].values.reshape(1, -1), 'max')[0]

    # 对词频排名
    tmp = tmp.sort_values(by='tf', ascending=False)
    tmp['tf_index'] = range(tmp.shape[0])

    # 对词跨度排名
    tmp = tmp.sort_values(by='word_distance_norm', ascending=False)
    tmp['word_distance_norm_index'] = range(tmp.shape[0])
    # tmp['word_distance_norm_index'] = normalize(tmp['word_distance_norm_index'].values.reshape(1, -1), 'max')[0]

    # 对Cosine排名
    tmp = tmp.sort_values(by='Cosine', ascending=False)
    tmp['cosine_index'] = range(tmp.shape[0])

    # 对EUC排名
    tmp = tmp.sort_values(by='Euclidean', ascending=True)
    tmp['euclidean_index'] = range(tmp.shape[0])

    # 如果是训练集，获取标签
    if train:
        true_entity = news['gt_entity'].split(',')
        # print(train_news['newsId'], true_entity, "文章长度为%d" % all_len)
        tmp['label'] = tmp['word'].apply(lambda w: int(w in true_entity))
        tmp = tmp.sort_values(by=['label', 'tfidf', 'tf'], ascending=False)
    else:
        tmp = tmp.sort_values(['tfidf', 'tf'], ascending=False)

    return tmp.reset_index(drop=True)


def get_train(df_, start_num, end_num, opt):
    # start_num = 0
    # end_num = 40000
    tfidf_model, w2v_model, d2v_model = load_models(df_)
    for i, num in enumerate(range(start_num, end_num)):
        print(i, "/", 10000, 'PID: ', int(end_num // 10000))
        tmp = extract_word_property(df_.iloc[num, :], tfidf_model, w2v_model, d2v_model, train=True)
        if i != 0:
            tmp.to_csv(os.path.join(os.path.join(opt['data_gen'], f'train_{start_num}_{end_num}.csv')), index=False, mode='a',
                       encoding='utf8',
                       header=False, float_format='%.6f')
        else:
            # print(i)
            tmp.to_csv(os.path.join(os.path.join(opt['data_gen'], f'train_{start_num}_{end_num}.csv')), index=False, mode='a',
                       encoding='utf8',
                       float_format='%.6f')

def get_test(df_, start_num, end_num, opt):
    # start_num = 0
    # end_num = 40000
    tfidf_model, w2v_model, d2v_model = load_models(df_)
    for i, num in enumerate(range(start_num, end_num)):
        print(i, "/", 10000, 'PID: ', int((end_num // 10000)))
        tmp = extract_word_property(df_.iloc[num, :], tfidf_model, w2v_model, d2v_model, train=False)
        if i != 0:
            tmp.to_csv(os.path.join(os.path.join(opt['data_gen'], f'test_{start_num}_{end_num}.csv')), index=False, mode='a',
                       encoding='utf8',
                       header=False, float_format='%.6f')
        else:
            # print(i)
            tmp.to_csv(os.path.join(os.path.join(opt['data_gen'], f'test_{start_num}_{end_num}.csv')), index=False, mode='a',
                       encoding='utf8',
                       float_format='%.6f')


def gen_train_and_test(opt):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%M-%d %H:%M:%S")

    stream_hanlder = logging.StreamHandler(sys.stdout)
    stream_hanlder.setFormatter(formatter)
    logger.addHandler(stream_hanlder)

    logger.info("正在加载所有分词结果...")
    df_ = pd.read_csv(os.path.join(opt['data_gen'], 'all_tokens.csv'))
    ## 加载实体字典
    ner_file = [f for f in os.listdir(ner_path) if f.endswith('txt')]

    for f in ner_file:
        jieba.load_userdict(os.path.join(ner_path, f))

    # train_data = loadData(train_path)

    #freeze_support()  # 这行代码用于windows平台
    p1 = mp.Process(target=get_train, args=(df_, 0, 20000, opt,))
    p2 = mp.Process(target=get_train, args=(df_, 20000, 40000,opt,))
    p3 = mp.Process(target=get_test, args=(df_, 40000, 60000,opt,))
    p4 = mp.Process(target=get_test, args=(df_, 60000, 80000,opt,))
    p5 = mp.Process(target=get_test, args=(df_, 80000, 100000,opt,))
    p6 = mp.Process(target=get_test, args=(df_, 100000, 120000,opt))
    # p7 = mp.Process(target=get_test, args=(df_,  100000, 110000,))
    # p8 = mp.Process(target=get_test, args=(df_,  110000, 120000,))

    # 开启进程
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    # p7.start()
    # p8.start()

    logger.info("开启全部进程...")

    # 关闭进程
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    #p7.join()
    # p8.join()

    logger.info("关闭全部进程...")
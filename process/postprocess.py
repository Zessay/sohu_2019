# -*- coding: utf-8 --*--
# @Author: Zessay
# @time: 2019.05.05 21:39
# @File: postprocess.py
# @Software: PyCharm

'''
选择性的对生成的文件进行一些后处理
'''

import os
import pandas as pd
import jieba.posseg as pseg

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from collections import Counter

from utils import loadData, bys
from config import DefaultConfig

def modify(df_, opt):
    ## 去除其中缺失单词的记录
    df_.dropna(axis=0, subset=['word'], inplace=True)
    ## 判断是否需要对词性进行修正
    if df_['cixing'].nunique() > 70:
        df_['cixing'] = df_['word'].apply(lambda w: next(pseg.cut(w)).flag)

    ## 去除训练集和测试集中的一些干扰值
    post_stopwords = []
    stop_path = os.path.join(opt['stopwords_dir'], opt['post_sw'])
    with open(stop_path, 'r', encoding='utf8') as f:
        for line in f:
            post_stopwords.append(line.strip())

    ## 过滤掉flag为1的值
    df_['flag'] = df_['word'].apply(lambda w: int(filter_num(w) or (w in post_stopwords)))
    df_ = df_[df_.flag == 0]
    df_.reset_index(drop=True, inplace=True)
    return df_

def filter_num(w):
    '''
    过滤一些训练集和测试集中的干扰字符
    :param w:
    :return:
    '''
    w = set(w)
    tar = [str(i) for i in range(10)] + ['.']
    for char in w:
        if char not in tar:
            return False
    return True

# --------------------------------------------------------------------------------------

def add_features(train_df, test_df, opt):
    '''
    增加新特征或者对特征进行转换
    :param train_df:
    :param test_df:
    :return:
    '''
    all_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    ## 对词性编码
    lb = LabelEncoder()
    all_df['cixing'] = lb.fit_transform(all_df['cixing'])
    ## 获取单词的逆词频
    all_df['idf'] = all_df['word'].map(Counter(all_df['word']))
    ## 获取训练集和测试集
    train_df = add_ID(all_df[:len(train_df)], opt)
    test_df = all_df[len(train_df):]

    return train_df, test_df

def add_ID(train_df, opt):
    path = os.path.join(opt['data_dir'], opt['train_file'])
    train_data = loadData(path)
    news2id = {}
    for i, news in enumerate(train_data):
        news2id[news['newsId']] = i
    df_ = pd.DataFrame({'newsId': list(news2id.keys()), 'ID': list(news2id.values())})
    train_df = train_df.merge(df_, on='newsId', how='left')
    return train_df

# ----------------------------------------------------------------------------------------

def gen_smooth_ctr(train_df, test_df=None, threshold=20, num=40000, opt=None):
    ## 获取训练集的文件
    path = os.path.join(opt['data_dir'], opt['train_file'])
    train_data = loadData(path)
    train_data = train_data[:num]
    ## 获取所有的真实实体
    entities = []
    for news in train_data:
        for item in news['coreEntityEmotions']:
            entities.append(item['entity'].strip())
    gt_all = Counter(entities)
    smooth_df = pd.DataFrame({'word': list(gt_all.keys()), 'num_in_gt': list(gt_all.values())})
    ## 搜索实体出现在训练集中多少个不同文档中
    occur_doc = {}
    print("\t统计实体在多少篇不同文章中出现...")
    for i, news in enumerate(train_data):
        text = news['title'] + '\n' + news['content']
        for gt in set(entities):
            if gt in text:
                occur_doc[gt] = occur_doc.get(gt, 0) + 1
        
        if i % 10000 == 0 and i != 0:
            print(f"\t\t已处理{i}篇文章")
    gt_occur = Counter(occur_doc)
    occur_df = pd.DataFrame({'word': list(gt_occur.keys()), 'occur_in_diff_doc': list(gt_occur.values())})
    
    ## 将两个DF合并
    smooth_df = smooth_df.merge(occur_df, on='word', how='left')
    ## 计算初始概率，由于一些实体在同一篇文章中被选中两次，需要进行修正
    smooth_df['raw_ratio'] = smooth_df['num_in_gt'] / smooth_df['occur_in_diff_doc']
    smooth_df.loc[smooth_df['raw_ratio'] > 1, 'num_in_gt'] = smooth_df[smooth_df['raw_ratio'] > 1]['occur_in_diff_doc'].values
    smooth_df['raw_ratio'] = smooth_df['num_in_gt'] / smooth_df['occur_in_diff_doc']
    ## 为了防止过拟合，只选取出现在不同文档中20次以上的实体进行平滑
    smooth_df = smooth_df[smooth_df['occur_in_diff_doc'] > threshold]

    ## 对概率值进行平滑
    print("\t正在进行转化率平滑操作...")
    ctr = bys(smooth_df['occur_in_diff_doc'].values, smooth_df['num_in_gt'].values)
    smooth_df['ctr'] = ctr

    if num < 40000:
        smooth_df.rename(columns={'num_in_gt': 'num_in_gt_35000',
                                  'occur_in_diff_doc': 'occur_in_diff_doc_35000',
                                  'ctr': 'ctr_35000'}, inplace=True)
    ## 将中间文件保存备用
    
    smooth_df.to_csv(os.path.join(opt['data_gen'], f'after_smooth_{threshold}_{num}.csv'), index=False)

    if num == 40000:
        train_df = train_df.merge(smooth_df[['word', 'num_in_gt', 'ctr']], on='word', how='left')
        test_df = test_df.merge(smooth_df[['word', 'num_in_gt', 'ctr']], on='word', how='left')
        return train_df, test_df
    else:
        train_df = train_df.merge(smooth_df[['word', 'num_in_gt_35000', 'ctr_35000']], on='word', how='left')
        return train_df
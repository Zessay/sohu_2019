# -*- coding: utf-8 --*--
# @Author: Zessay
# @time: 2019.05.05 19:40
# @File: generate_all_tokens.py
# @Software: PyCharm

import jieba
import jieba.posseg as pseg
import pandas as pd
import gc
import os
import re
import codecs
import copy

from tqdm import tqdm

from utils import *
from .train_models import train_kmeans, train_lda
from config import DefaultConfig


def generate_tokens(opt):
    '''
    加载训练集和测试集的文件，生成训练集和测试集的拼接，包含全部文章的title，content等内容
    '''
    ## 加载训练集和测试集数据
    train_data = loadData(os.path.join(opt['data_dir'], opt['train_file']))
    test_data = loadData(os.path.join(opt['data_dir'], opt['test_file']))

    ## 将训练集和测试集加载为df
    print("正在加载训练集...")
    train_df = pd.DataFrame()
    for i in tqdm(range(len(train_data))):
        tmp = pd.DataFrame(train_data[i])
        train_df = pd.concat([train_df, tmp], axis=0, ignore_index=True)

    print("正在加载测试集...")
    test_df = pd.DataFrame()
    for item in tqdm(test_data):
        # print(item)
        tmp = pd.DataFrame(item, index=[0])
        test_df = pd.concat([test_df, tmp], axis=0, ignore_index=True)

    ## 将两组数据进行合并
    print("合并训练集和测试集...")
    all_df = pd.concat([train_df[['newsId', 'title', 'content']].drop_duplicates(), test_df], axis=0, ignore_index=True)

    del test_data, train_df, test_df
    gc.collect()

    ## 清理一些特殊字符
    all_df['title'] = all_df['title'].apply(lambda s: process_text(s))
    all_df['content'] = all_df['content'].apply(lambda s: process_text(s))
    all_df['text'] = all_df['title'] + '\n' + all_df['content']

    ## 加载bert的结果
    bert_train = []

    with open('result/bert10train.txt', 'r', encoding='utf8') as f:
        for line in f:
            line = line.replace('#', '')
            line = re.sub(',\[unk\],', ',', line)
            bert_train.append(line.strip())

    bert_test = []
    with open('result/bert10test.txt', 'r', encoding='utf8') as f:
        for line in f:
            line = line.replace('#', '')
            line = re.sub(',\[unk\],', ',', line)
            bert_test.append(line.strip())


    bert_all = bert_train + bert_test
    all_df['bert_result'] = bert_all

    ## 加载停止词词典
    rawwords = loadStopWords(os.path.join(opt['stopwords_dir'],opt['simple_sw']))
    postwords = loadStopWords(os.path.join(opt['stopwords_dir'],opt['post_sw']))

    stopwords = rawwords + postwords
    all_df['title_cut'] = all_df['title'].apply(lambda s: ','.join([word.strip() for word in jieba.cut(s) if (word.strip() not in stopwords and word.strip())]))

    title = []
    for i in tqdm(range(all_df.shape[0])):
        news = all_df.iloc[i, :]
        result = ngram(news)
        title.append(result)
    all_df['title_ngram'] = title

    ## 加载分词需要使用的实体字典
    #loadDict(opt)

    if not os.path.isfile(os.path.join(opt['data_gen'], 'train_ent_emo.csv')):
        extract_ent_emo(train_data)

    train_ent = pd.read_csv(os.path.join(opt['data_gen'], 'train_ent_emo.csv'))

    all_df = all_df.merge(train_ent, on='newsId', how='left')


    train_df = all_df[:len(train_data)]
    test_df = all_df[len(train_data):]

    train_df['bert_and_title'] = train_df['bert_result'] + ',' + train_df['gt_entity'] + ',' + train_df['title_ngram']
    test_df['bert_and_title'] = test_df['bert_result'] + ',' + test_df['title_ngram']

    all_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)

    all_df['tokens'] = all_df['bert_and_title'].apply(
        lambda s: ','.join([w.strip() for w in set(s.split(',')) if w and len(w.strip()) > 1]))

    all_df['cixing'] = all_df['tokens'].apply(lambda s: get_cixing(s))

    all_df.drop(['bert_result', 'title_cut', 'title_ngram', 'bert_and_title'], axis=1, inplace=True)

    all_df['tokens_with_sw'] = all_df['text'].apply(lambda s: '/split'.join(jieba.cut(s)))
    all_df['tokens_with_sw'] = add_bert_to_tokens(all_df)

    if os.path.isfile('data/final_classes.csv'):
        final_classes = pd.read_csv('data/final_classes.csv')
        all_df = all_df.merge(final_classes, on='newsId', how='left')
    else:
        ## 训练并获取聚类结果和主题模型结果
        all_df['lda_classes'] = train_lda(all_df, opt)
        all_df['kmeans_classes'] = train_kmeans(all_df, opt)
    ## 保存文件
    all_df.to_csv(os.path.join(opt['data_gen'], 'all_tokens.csv'), index=False)


def add_bert_to_tokens(all_df):
    result = []
    for i in tqdm(range(all_df.shape[0])):
        news = all_df.iloc[i,:]
        tokens = news['tokens'].split(',')
        all_tokens = news['tokens_with_sw']
        for word in tokens:
            if word not in all_tokens:
                all_tokens = '/split' + word + all_tokens + '/split' + word
        result.append(all_tokens)

    return result


def process_text(text):
    '''
    定义对文本的特殊字符进行处理的函数
    '''
    pattern = [(r'&amp;', '&'), (r'&lt;', '<'), (r'&gt;', '>'), (r'&quot;', ''),
           (r'&nbsp;', ''), (r'br/', ''), (r'&', '&')]

    # 根据pattern清除一些特殊字符
    for p in pattern:
        text = re.sub(p[0], p[1], text)

    # 将内容中一些明显的url字符清除
    ## html1是以http开头的url
    ## html2是以www开头的url
    html1 = re.compile(r'(https?://)([\da-z\.-]+)\.([a-z\.]{2,6})([/\w \.-]*)*/?')
    html2 = re.compile(r'(www)\.([\da-z\.-]+)\.([a-z\.]{2,6})([/\w \.-]*)*/?')

    text = re.sub(html1, ' ', text)
    text = re.sub(html2, ' ', text)

    return text


# 过滤干扰字符
def filter_num(w):
    w = set(w)
    tar = [str(i) for i in range(10)] + ['.']
    for char in w:
        if char not in tar:
            return False
    return True


def loadDict(opt):
    # 加载分词需要使用的实体字典
    nerdict = [f for f in os.listdir(opt['data_dict']) if f.endswith('txt')]
    for file in nerdict:
        jieba.load_userdict(os.path.join(opt['data_dict'], file))


def loadStopWords(path):
    stopwords = []
    # 加载停止词
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            stopwords.append(line.strip())
    return stopwords


def ngram(news):
    fenci = news['title_cut'].split(',')
    text = news['title'] + '\n' + news['content']
    result = copy.deepcopy(fenci)
    length = len(fenci)
    appear = []
    for i in range(length):
        if i < length - 1:
            combine = fenci[i] + fenci[i + 1]
            count = text.count(combine)
            if count > 1 and combine not in appear:
                appear.append(combine)
                result.append(combine)

            combine = fenci[i] + ' ' + fenci[i + 1]
            count = text.count(combine)
            if count > 1 and combine not in appear:
                appear.append(combine)
                result.append(combine)

            if i < length - 2:
                ## 分词合并
                combine = fenci[i] + fenci[i + 1] + fenci[i + 2]
                count = text.count(combine)
                if count > 1 and combine not in appear:
                    appear.append(combine)
                    result.append(combine)

                ## 分词加空格合并
                combine = fenci[i] + ' ' + fenci[i + 1] + fenci[i + 2]
                count = text.count(combine)
                if count > 1 and combine not in appear:
                    appear.append(combine)
                    result.append(combine)

                ## 分词加空格合并
                combine = fenci[i] + fenci[i + 1] + ' ' + fenci[i + 2]
                count = text.count(combine)
                if count > 1 and combine not in appear:
                    appear.append(combine)
                    result.append(combine)
    return ','.join(result)


def get_cixing(s):
    tokens = s.split(',')
    cixing = []
    for word in tokens:
        try:
            p = next(pseg.cut(word)).flag
            cixing.append(p)
        except:
            print(word, s)

    assert len(tokens) == len(cixing), '长度不匹配'
    return ','.join(cixing)
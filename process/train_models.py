# -*- coding: utf-8 --*--
# @Author: Zessay
# @time: 2019.05.05 20:21
# @File: train_models.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

from config import DefaultConfig

'''
# 读取生成的分词文件给后面模型训练使用
token_path = '../data/gen/all_tokens.csv'
all_df = pd.read_csv(token_path)
'''

# 定义训练TFIDF的模型
def train_tfidf(all_df):
    opt = DefaultConfig()
    tfidf_tokens = all_df['tokens'].values
    tfidf_model = TfidfVectorizer()
    print("正在训练TFIDF模型...")
    tfidf_model = tfidf_model.fit(tfidf_tokens)

    with open(os.path.join(opt['model_word'], 'Tfidf.pkl'), 'wb') as f:
        pickle.dump(tfidf_model, f)
    print("TFIDF模型已保存...")

# 定义训练word2vec的模型
def train_word2vec(all_df):
    opt = DefaultConfig()
    sentences = [line.strip().split('/split') for line in all_df['tokens_with_sw']]
    # 训练200维的词向量
    word2vec = Word2Vec(size=200, window=5, min_count=1, iter=20)
    word2vec.build_vocab(sentences)
    print("正在训练Word2Vec模型...")
    word2vec.train(sentences, total_examples=word2vec.corpus_count, epochs=word2vec.epochs)
    word2vec.save(os.path.join(opt['model_word'], 'word2vec.model'))
    print("Word2Vec模型已保存...")

# 定义训练doc2vec的模型
def train_doc2vec(all_df):
    opt = DefaultConfig()
    newsid = all_df['newsId']
    sentences = [line.strip().split('/split') for line in all_df['tokens_with_sw']]
    # 构建文章向量的语料
    documents = [TaggedDocument(doc, [ID]) for doc, ID in zip(sentences, newsid)]
    doc2vec = Doc2Vec(vector_size=200, window=5, min_count=1, epochs=20)
    doc2vec.build_vocab(documents)
    print("正在训练Doc2Vec模型...")
    doc2vec.train(documents, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)
    doc2vec.save(os.path.join(opt['model_word'], 'doc2vec.model'))
    print("Doc2Vec模型已保存...")

# 训练LDA主题分类模型
def train_lda(all_df, opt, n_topics=10):
    ## 使用不包含停止词的分词结果
    corpus = all_df['tokens']
    cnt = CountVectorizer()
    cntIf = cnt.fit_transform(corpus)

    lda_path = os.path.join(opt['model_word'], 'lda.pkl')
    try:
        with open(lda_path, 'rb') as f:
            lda_model = pickle.load(f)
        lda_pred = lda_model.transform(cntIf)
        lda_classes = np.argmax(lda_pred, axis=1)
    except:
        ## 使用LDA主题模型进行分类
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=500)
        print("正在训练LDA主题模型...")
        lda_pred = lda.fit_transform(cntIf)
        lda_classes = np.argmax(lda_pred, axis=1)
        ## 保存模型
        with open(lda_path, 'wb') as f:
            pickle.dump(lda, f)
        print("LDA主题模型已保存...")

    return lda_classes

# 训练KMeans分类器
def train_kmeans(all_df, opt, n_clusters=10):
    ## 准备训练集
    cluster_train = []
    newsid = all_df['newsId']
    doc2vec = Doc2Vec.load(os.path.join(opt['model_word'], 'doc2vec.model'))

    for ID in newsid:
        cluster_train.append(doc2vec[ID])

    kmeans_path = os.path.join(opt['model_word'], 'kmeans.pkl')

    try:
        with open(kmeans_path, 'rb') as f:
            kmeans_model = pickle.load(f)
        kmeans_classes = kmeans_model.predict(cluster_train)
    except:
        cluster = KMeans(n_clusters=n_clusters)
        print("正在训练KMeans聚类结果...")
        kmeans_classes = cluster.fit_predict(cluster_train)
        ## 保存模型
        with open(kmeans_path, 'wb') as f:
            pickle.dump(cluster, f)
        print("KMean聚类模型已保存...")

    return kmeans_classes

def train_all_models(opt):
    all_df = pd.read_csv(os.path.join(opt['data_gen'], 'all_tokens.csv'))
    # 训练TFIDF
    train_tfidf(all_df)
    # 训练word2vec
    train_word2vec(all_df)
    # 训练doc2vec
    train_doc2vec(all_df)
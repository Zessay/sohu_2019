# -*- coding: utf-8 --*--
# @Author: Zessay
# @time: 2019.05.05 19:28
# @File: threshold.py
# @Software: PyCharm

'''
阈值搜索函数：
find_threshold 用于验证集搜索阈值
return_entity 用于按照搜索得到的阈值返回预测实体
'''
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
from config import DefaultConfig
import os
from .subsave import *

## 根据验证集搜索阈值
def find_threshold(y_true, y_pred, y_probability, n=4):
    '''
    y_true: 真实实体，二维列表，第一层每一个元素表示一篇文章，第二层表示文章中的实体
    y_pred: 预测实体，二维列表

    y_probability: 预测实体的概率，二维列表
    n: 表示要得到的实体的数量
    '''
    best_threshold = 0
    best_score = -1
    for threshold1 in [0]:
        for threshold2 in tqdm([i*0.1 for i in range(3, 8)]):
            for threshold3 in [i*0.1 for i in range(3, 8)]:
                for threshold4 in [i*0.1 for i in range(4, 9)]:
                    for threshold5 in [i*0.1 for i in range(2, 8)]:
                        for threshold6 in [i*0.1 for i in range(2, 8)]:
                            for threshold7 in [i*0.1 for i in range(2, 8)]:
                                # 计算每一个阈值的f1得分
                                pred_metric = []
                                true_metric = []
                                # 对于每一个真实实体，预测实体和对应的概率
                                for true, pred, proba in zip(y_true, y_pred, y_probability):
                                    # print(true, end=' ')
                                    ents = []
                                    Y_ents = []
                                    c = 0
                                    # 对于最大的概率值
                                    tops = proba[0]
                                    # 对于每一个预测实体以及其对应的概率
                                    for ped, proba in zip(pred, proba):
                                        # print(proba, end=' ')
                                        if c == n:
                                            break
                                        if c == 0:
                                            if proba < threshold1:
                                                break
                                        if c == 1:
                                            if proba < tops*threshold2 or proba<threshold5:
                                                break
                                        if c == 2:
                                            if proba < tops*threshold3 or proba<threshold6:
                                                break
                                        if c == 3:
                                            if proba < tops*threshold4 or proba<threshold7:
                                                break

                                        # 否则，将该实体添加到真实实体中
                                        ents.append(ped)
                                        c += 1
                                    # print(ents)
                                    # 对于所有的真实实体
                                    for y in true:
                                        Y_ents.append(y)

                                    # 模型评估
                                    ## 对于预测实体中的每一个
                                    for e in ents:
                                        ## 如果在真实实体中
                                        if e in Y_ents:  ## TP
                                            pred_metric.append(1)
                                            true_metric.append(1)
                                        else:  ## FP
                                            pred_metric.append(1)
                                            true_metric.append(0)
                                            ## 计算FN
                                    for y in Y_ents:
                                        if y not in ents:
                                            pred_metric.append(0)
                                            true_metric.append(1)

                                score = f1_score(true_metric, pred_metric)
                                if score > best_score:
                                    best_threshold1 = threshold1
                                    best_threshold2 = threshold2
                                    best_threshold3 = threshold3
                                    best_threshold4 = threshold4
                                    best_threshold5 = threshold5
                                    best_threshold6 = threshold6
                                    best_threshold7 = threshold7
                                    best_score = score
        print('tmp best f1-score: ', best_score)
    search_result = {'best_f1_score': best_score}  # 记录最好的f1值
    thresholds = {
        'best_threshold1': best_threshold1,
        'best_threshold2': best_threshold2,
        'best_threshold3': best_threshold3,
        'best_threshold4': best_threshold4,
        'best_threshold5': best_threshold5,
        'best_threshold6': best_threshold6,
        'best_threshold7': best_threshold7
    }
    print(search_result)
    print(thresholds)
    return thresholds, search_result


## 用于根据搜索阈值返回预测结果
def return_entity(y_pred, y_probability, th, n=4):
    '''
    y_pred：一个二维列表，第一维表示每一篇文章，第二维表示文章中的每一个单词
    y_probability：二维列表，对应每一个单词的预测概率

    '''
    # 对于每一个真实实体，预测实体和对应的概率
    entities = []
    for pred, proba in zip(y_pred, y_probability):
        # print(true, end=' ')
        ents = []
        c = 0
        # 对于最大的概率值
        tops = proba[0]
        # 对于每一个预测实体以及其对应的概率
        for ped, proba in zip(pred, proba):
            # print(proba, end=' ')
            if c == n:
                break
            if c == 0:
                if proba < th['best_threshold1']:
                    break
            if c == 1:
                if proba < tops * th['best_threshold2'] or proba < th['best_threshold5']:
                    break
            if c == 2:
                if proba < tops * th['best_threshold3'] or proba < th['best_threshold6']:
                    break
            if c == 3:
                if proba < tops * th['best_threshold4'] or proba < th['best_threshold7']:
                    break

            # 否则，将该实体添加到真实实体中
            ents.append(ped)
            c += 1
        entities.append(','.join(ents))
    return entities

# ----------------------------------------------------------------------------------------------

def sort_val(val_, y_pred, predict=False, name='lgb_pred'):
    '''
    对y_pred的结果进行排序，获取训练集的实体
    :param val_:
    :param y_pred:
    :param name:
    :return:
    '''
    val = val_[['word', 'newsId']]
    val[name] = y_pred
    ## 获取不同新闻的预测值并排序
    pred_id = []
    pred_words = []
    pred_proba = []
    for ID, group in val.groupby('newsId'):
        gp = group.sort_values(by=[name], ascending=False)
        pred_id.append(ID)
        pred_words.append(gp['word'].values)
        pred_proba.append(gp[name].values)

    if not predict:
        gt_ent = get_gt(pred_id)
        return gt_ent, pred_words, pred_proba
    else:
        return pred_id, pred_words, pred_proba

def get_gt(pred_id):
    # 定义配置文件对象
    opt = DefaultConfig()
    val_id = pd.DataFrame({'newsId': pred_id})
    ## 打开训练集的Id的顺序以及其对应的单词
    train_ent_path = os.path.join(opt['data_gen'], 'train_ent_emo.csv')
    
    if not os.path.isfile(train_ent_path):
        train_path = os.path.join(opt['data_dir'], opt['train_file'])
        train_data = loadData(train_path)
        extract_ent_emo(train_data)
    
    train_ent = pd.read_csv(train_ent_path)
    val = val_id.merge(train_ent[['newsId', 'entity']], on='newsId', how='left')
    ## 得到训练集的所有实体
    gt_ent = list(map(lambda x: x.split(','), val['entity'].values))
    return gt_ent

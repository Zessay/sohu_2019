# -*- coding: utf-8 --*--
# @Author: Zessay
# @time: 2019.05.05 19:26
# @File: subsave.py
# @Software: PyCharm

'''
用于按顺序生成提交文件以及加载源文件
'''

import pandas as pd
import json
import pickle
import os

# 定义用于按顺序保存提交文件的函数
def save_as_order(id, entity, opt, name):
    '''
    id: 表示预测结果的id
    entity: 表示预测实体
    opt: 表示配置对象
    name: 表示保存的文件名
    '''
    result = pd.DataFrame()
    result['newsId'] = id
    result['entity'] = entity

    ## 读取参照顺序的源文件
    test_path = os.path.join(opt['data_gen'], 'test_id.csv')
    if not os.path.isfile(test_path):
        gen_testid(opt)
    true = pd.read_csv(test_path)
    result = true.merge(result, on='newsId', how='left')
    with open(os.path.join(opt['result_dir'], name), 'w', encoding='utf8') as f:
        for newsId, entity in zip(result['newsId'].values, result['entity'].values):
            submit = ['{}\t{}\t{}\n'.format(newsId, entity, ','.join(['POS'] * len(entity.split(','))))]
            f.writelines(submit)


## ------------------------------------------------------------------------------------------------

# 加载原始的训练集的文件
def loadData(path):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))

    return data


## ------------------------------------------------------------------------------------------------

# 加载生成的提交文件
def read_sub_file(path):
    test_id = []
    test_ent = []
    test_emo = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            record = line.split('\t')
            test_id.append(record[0])
            test_ent.append(record[1])
    return test_id, test_ent


## ------------------------------------------------------------------------------------------------


def extract_ent_emo(train_data, save=True):
    '''
    # 从train_data中提取id以及对应的实体和情感
    train_data: 列表型，每一个元素是一个news
    '''
    train_id = []
    train_ent = []
    train_emo = []

    for news in train_data:
        train_id.append(news['newsId'])
        ent = []
        emo = []
        for item in news['coreEntityEmotions']:
            ent.append(item['entity'].strip())
            emo.append(item['emotion'])
        train_ent.append(','.join(ent))
        train_emo.append(','.join(emo))

    if save:
        train_ent_emo = pd.DataFrame(
            {
                'newsId': train_id,
                'entity': train_ent,
                'emotion': train_emo
            }
        )
        train_ent_emo.to_csv('data/gen/train_ent_emo.csv', index=False)
    return train_id, train_ent, train_emo

## -------------------------------------------------------------------------------------------

def save_pkl(model, path):
    '''
    将模型保存为文件
    :param model:
    :param path:
    :return:
    '''
    with open(path, 'wb') as f:
        pickle.dump(model, f)


## --------------------------------------------------------------------------------------------

def gen_testid(opt):
    test_path = os.path.join(opt['data_dir'], opt['test_file'])
    test_data = loadData(test_path)

    test_id = []
    for news in test_data:
        test_id.append(news['newsId'])

    test_ = pd.DataFrame({'newsId': test_id})
    test_.to_csv(os.path.join(opt['data_gen'], 'test_id.csv'), index=False)
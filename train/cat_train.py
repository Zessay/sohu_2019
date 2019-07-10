# -*- coding: utf-8 --*--
# @Author: Zessay
# @time: 2019.05.09 9:49
# @File: cat_train.py
# @Software: PyCharm

import os
import catboost
from catboost import Pool
from utils import sort_val, find_threshold, save_pkl, return_entity, save_as_order
from config import DefaultConfig


class CATTrain:
    def __init__(self, train, test, params):
        self.train_ = train
        self.test_ = test
        self.params = params
        self.threshold = None
        self.model = None
        self.fe = None
        self.pred = None
        self.opt = DefaultConfig()

    # 训练部分训练集数据，在验证集搜索分割阈值
    def train_for_threshold(self, features, target='label', num=35000):
        train_df = self.train_[self.train_.ID < num]
        val_df = self.train_[self.train_.ID >= num]

        X_train, y_train = train_df[features].values, train_df[target].values.astype('uint8')
        X_eval, y_eval = val_df[features].values, val_df[target].values.astype('uint8')

        cat_train = Pool(X_train, y_train)
        cat_eval = Pool(X_eval, y_eval)

        cat_model = catboost.train(cat_train, self.params, iterations=10000,
                              eval_set=cat_eval,
                              early_stopping_rounds=200,
                              verbose=500)
        y_pred = cat_model.predict(cat_eval, prediction_type='Probability')[:,1]
        ## 获取验证集的真实实体，以及按顺序排序预测的概率和对应的单词
        gt_ent, pred_words, pred_proba = sort_val(val_df, y_pred)
        ## 获取搜索得到的阈值结果
        self.threshold, _ = find_threshold(gt_ent, pred_words, pred_proba)

        return self.threshold

    # 训练全部数据
    def train_and_predict(self, features, target='label', save=True):
        self.fe = features
        X_train, y_train = self.train_[features].values, self.train_[target].values.astype('uint8')
        X_test = self.test_[self.fe].values

        cat_all = Pool(X_train, y_train)

        model = catboost.train(
            cat_all,
            self.params,
            iterations=10000,
            early_stopping_rounds=200,
            verbose=1000
        )

        self.model = model

        if save:
            save_pkl(model, os.path.join(self.opt['model_train'], 'cat.pkl'))

        self.pred = model.predict(X_test, prediction_type='Probability')[:, 1]

        return self.pred

    # 保存测试集得到的结果
    def save_result(self):
        pred_id, pred_words, pred_proba = sort_val(self.test_, self.pred, predict=True)
        entities = return_entity(pred_words, pred_proba, self.threshold)

        save_as_order(pred_id, entities, self.opt, 'cat_result.txt')







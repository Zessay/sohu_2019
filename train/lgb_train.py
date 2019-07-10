# -*- coding: utf-8 --*--
# @Author: Zessay
# @time: 2019.05.06 9:33
# @File: lgb_train.py
# @Software: PyCharm

import numpy as np
import os

from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from utils import sort_val, find_threshold, save_pkl, return_entity, save_as_order

from config import DefaultConfig

class LGBTrain:
    def __init__(self, train, test, opt):
        self.train_ = train
        self.test_ = test
        self.params = opt['lgb_params']
        self.threshold = None
        self.model = None
        self.fe = None
        self.pred = None
        self.opt = opt

    # 训练部分训练集数据，在验证集搜索分割阈值
    def train_for_threshold(self, features, target='label', num=35000):
        train_df = self.train_[self.train_.ID < num]
        val_df = self.train_[self.train_.ID >= num]

        X_train, y_train = train_df[features].values, train_df[target].values.astype('uint8')
        X_eval, y_eval = val_df[features].values, val_df[target].values.astype('uint8')

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        lgb_model = lgb.train(self.params, lgb_train, num_boost_round=10000,
                              valid_sets=[lgb_train, lgb_eval],
                              valid_names=['train', 'valid'],
                              early_stopping_rounds=100,
                              verbose_eval=1000)
        y_pred = lgb_model.predict(X_eval)
        ## 获取验证集的真实实体，以及按顺序排序预测的概率和对应的单词
        gt_ent, pred_words, pred_proba = sort_val(val_df, y_pred)
        ## 获取搜索得到的阈值结果
        self.threshold, _ = find_threshold(gt_ent, pred_words, pred_proba)

        return self.threshold

    # 训练全部数据
    def train_and_predict(self, features, target='label', n_folds=5, save=True, seed=2019):
        self.fe = features
        X_train, y_train = self.train_[features].values, self.train_[target].values.astype('uint8')
        X_test = self.test_[self.fe].values

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        self.pred = np.zeros(len(X_test))

        ## 进行K折训练
        for fold_, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            train_X, train_y = X_train[train_idx], y_train[train_idx]
            eval_X, eval_y = X_train[val_idx], y_train[val_idx]

            lgb_train = lgb.Dataset(train_X, train_y)
            lgb_eval = lgb.Dataset(eval_X, eval_y)

            print('\nFold: ', fold_)
            model = lgb.train(
                self.params,
                lgb_train,
                num_boost_round=7000,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=['train', 'eval'],
                early_stopping_rounds=200,
                verbose_eval=1000
            )

            if save:
                save_pkl(model, os.path.join(self.opt['model_train'], f'lgb_fold_{fold_}.pkl'))

            self.pred += model.predict(X_test) / n_folds
        return self.pred


    # 保存测试集得到的结果
    def save_result(self):
        pred_id, pred_words, pred_proba = sort_val(self.test_, self.pred, predict=True)
        entities = return_entity(pred_words, pred_proba, self.threshold)

        save_as_order(pred_id, entities, self.opt, 'lgb_result.txt')







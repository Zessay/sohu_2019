# -*- coding: utf-8 --*--
# @Author: Zessay
# @time: 2019.05.09 9:41
# @File: xgb_train.py
# @Software: PyCharm

import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from utils import sort_val, find_threshold, save_pkl, return_entity, save_as_order

from config import DefaultConfig


class XGBTrain:
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

        xgb_train = xgb.DMatrix(X_train, y_train)
        xgb_eval = xgb.DMatrix(X_eval, y_eval)

        xgb_model = xgb.train(self.params, xgb_train, num_boost_round=1000,
                              evals=[(xgb_train, 'train'), (xgb_eval, 'eval')],
                              early_stopping_rounds=100,
                              verbose_eval=100)
        y_pred = xgb_model.predict(xgb_eval)
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
        xgb_test = xgb.DMatrix(X_test)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        self.pred = np.zeros(len(X_test))

        ## 进行K折训练
        for fold_, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            train_X, train_y = X_train[train_idx], y_train[train_idx]
            eval_X, eval_y = X_train[val_idx], y_train[val_idx]

            xgb_train = xgb.DMatrix(train_X, train_y)
            xgb_eval = xgb.DMatrix(eval_X, eval_y)

            print('\nFold: ', fold_)
            model = xgb.train(
                self.params,
                xgb_train,
                num_boost_round=500,
                evals=[(xgb_train, 'train'), (xgb_eval, 'eval')],
                early_stopping_rounds=100,
                verbose_eval=100
            )

            if save:
                save_pkl(model, os.path.join(self.opt['model_train'], f'xgb_fold_{fold_}.pkl'))

            self.pred += model.predict(xgb_test) / n_folds
        return self.pred


    # 保存测试集得到的结果
    def save_result(self):
        pred_id, pred_words, pred_proba = sort_val(self.test_, self.pred, predict=True)
        entities = return_entity(pred_words, pred_proba, self.threshold)

        save_as_order(pred_id, entities, self.opt, 'xgb_result.txt')







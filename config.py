# -*- coding: utf-8 --*--
# @Author: Zessay
# @time: 2019.05.09 10:01
# @File: config.py
# @Software: PyCharm

class DefaultConfig(dict):
    def __init__(self):
        # 用于存储模型、数据、停止词以及结果的路径
        self['data_dir'] = 'data'
        self['data_dict'] = 'data/nerdict'
        self['data_gen'] = 'data/gen'

        ## 用于保存模型和结果的位置
        self['model_word'] = 'models/word'
        self['model_train'] = 'models/machine'
        self['result_dir'] = 'result'

        ## 停止词相关的路径
        self['stopwords_dir'] = 'stopwords'
        self['post_sw'] = 'post_stopwords.txt'
        self['simple_sw'] = 'simple_stopwords.txt'
        self['special_sw'] = 'special_stopwords.txt'

        ## 训练集和测试集的文件名
        self['train_file'] = 'coreEntityEmotion_train.txt'
        self['test_file'] = 'coreEntityEmotion_test_stage2.txt'

        # 3个模型训练的参数
        self['lgb_params'] = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'xentropy',

            'num_leaves': 63,
            'learning_rate': 0.01,

            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_seed': 0,
            'bagging_freq': 1,
            'verbose': 1,
            'reg_alpha': 1,
            'reg_lambda': 2,

            'seed': 2019,

            # 设置GPU
            'device': 'gpu',
            'gpu_platform_id': 1,
            'gpu_device_id': 0
        }

        self['xgb_params'] = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',

            'eval_metric': 'logloss',

            'learning_rate': 0.0894,
            'max_depth': 9,
            'max_leaves': 20,

            'lambda': 2,
            'alpha': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'silent': 1,
            'seed': 2019,

            # 使用gpu
            'gpu_id': 0,
            'tree_method': 'gpu_hist'
        }

        self['cat_params'] = {
            'loss_function': 'Logloss',
            'eval_metric': 'F1',

            'learning_rate': 0.05,
            'max_depth': 5,
            'max_leaves_count': 63,

            'reg_lambda': 2,

            'verbose': 1,
            'random_seed': 2019,

            # 使用GPU
            'od_type': 'Iter',
            'task_type': 'GPU'
        }
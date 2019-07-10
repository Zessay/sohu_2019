from process import *
from train import *
from utils import *
from config import DefaultConfig
import os
import pandas as pd
import numpy as np

def main():
    opt = DefaultConfig()
    generate_all_tokens(opt)
    gen_train_and_test(opt)

    train_file = [f for f in os.listdir(opt['data_gen']) if f.startswith('train') and f.endswith('00.csv')]
    test_file = [f for f in os.listdir(opt['data_gen']) if f.startswith('test') and f.endswith('00.csv')]

    train_df = pd.DataFrame()
    for file in train_file:
        tmp = pd.read_csv(os.path.join(opt['data_gen'], file))
        train_df = pd.concat([train_df, tmp], axis=0, sort=False, ignore_index=True)

    test_df = pd.DataFrame()
    for file in test_file:
        tmp = pd.read_csv(os.path.join(opt['data_gen'], file))
        test_df = pd.concat([test_df, tmp], axis=0, sort=False, ignore_index=True)

    train_df, test_df = add_features(train_df, test_df, opt)
    train_df, test_df = gen_smooth_ctr(train_df=train_df, test_df=test_df, threshold=10, opt=opt)
    train_df = gen_smooth_ctr(train_df=train_df, threshold=10, num=40000, opt=opt)
    train_df = add_ID(train_df)

    lgb_train = LGBTrain(train_df, test_df, opt)

    lgb_fe_1 = [f for f in train_df.columns if f not in (['word', 'newsId', 'num_in_gt',
                                                          'ctr', 'ID', 'label', ])]
    lgb_fe_2 = [f for f in train_df.columns if f not in (['word', 'newsId', 'num_in_gt_35000',
                                                          'ctr_35000', 'ID', 'label', ])]

    threshold = lgb_train.train_for_threshold(lgb_fe_1)

    y_pred = lgb_train.train_and_predict(lgb_fe_2)
    lgb_train.save_result()


if __name__ == "__main__":
    main()
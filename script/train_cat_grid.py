#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import GridSearchCV
from itertools import product
import time

MODEL_NUM = 10
NUM_ENSEMBLE = 5

cat_feature_inds = [0,1,2]+[i for i in range(12,78)]+[79,80,81,84,85,86]+[i for i in range(124,204)]


def normalize_data(org_data):
    data = org_data.dropna()
    data = data.reset_index()
    return data


def get_data_from_csv(begin_date, end_date, fname_csv, course=0, kind=0, nData=47):
    df = pd.read_csv(fname_csv)
    remove_index = []
    for idx in range(len(df)):
        date = int(df['date'][idx])
        if date < begin_date or date > end_date or (course > 0 and course != int(df['course'][idx])) or (kind > 0 and kind != int(df['kind'][idx])):
            remove_index.append(idx)
    data = df.drop(df.index[remove_index])
    data = normalize_data(data)

    R_data = data[['name', 'rank', 'r1', 'r2', 'r3', 'hr_nt', 'hr_dt', 'jk_nt', 'tr_nt', 'cnt', 'rcno', 'price', 'bokyeon1', 'bokyeon2', 'bokyeon3', 'boksik', 'ssang', 'sambok', 'samssang', 'idx']]
    Y_data = data['rctime']
    X_data = data.copy()
    X_data = X_data.drop(['name', 'rctime', 'rank', 'r3', 'r2', 'r1', 'date', 'price', 'bokyeon1', 'bokyeon2', 'bokyeon3', 'boksik', 'ssang', 'sambok', 'ssang', 'samssang', 'index'], axis=1)
    return X_data, Y_data, R_data, data


def parameter_tunning(train_bd, train_ed):
    train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
    train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))

    test_bd = train_ed + datetime.timedelta(days=7)
    test_ed = train_ed + datetime.timedelta(days=7+30)
    test_bd_i = int("%d%02d%02d" % (test_bd.year, test_bd.month, test_bd.day))
    test_ed_i = int("%d%02d%02d" % (test_ed.year, test_ed.month, test_ed.day))

    print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
    X_train, Y_train, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016_v1.csv')
    X_test, Y_test, _, _ = get_data_from_csv(test_bd_i, test_ed_i, '../data/1_2007_2016_v1.csv')

    train_pool = Pool(X_train, Y_train, cat_features=cat_feature_inds)
    test_pool = Pool(X_test, Y_test, cat_features=cat_feature_inds)

    print("Done")

    for itr in [500,700,1000,1500]:
        for lr in [4,5,6,7,8]:
            for dep in [3,4,5]:
                for llr in [1,2,3]:
                    print("SETUP: iter: %5d, lr: %2.0f, depth: %2d, l2_leaf_reg: %2d" % 
                        (itr, lr, dep, llr), end='\t')
                    scores = []
                    # Use CV
                    clf = CatBoostRegressor(iterations=itr,
                                            learning_rate=lr,
                                            depth=dep, 
                                            l2_leaf_reg=llr,
                                            thread_count=50,
                                            loss_function='MAE',
                                            eval_metric='MAE')
                    model_path = '../model/catboost/i%d_lr%d_dep%d_llr%d/%d_%d' % (itr, lr, dep, llr, train_bd_i, train_ed_i)
                    if os.path.exists("%s/model.pkl" % model_path):
                        clf.load_model("%s/model.pkl" % model_path)
                    else:
                        time1 = time.time()
                        clf.fit(train_pool)
                        time2 = time.time()
                        print("time took %.3f s" % ((time2-time1),), end='\t')
                        if (time2-time1) > 120:
                            if not os.path.exists("%s" % model_path):
                                os.makedirs("%s" % model_path)
                            clf.save_model("%s/model.pkl" % model_path)
                    print("score train: %6.0f, test: %6.0f" % (clf.score(train_pool, Y_train), clf.score(test_pool, Y_test)))


if __name__ == '__main__':
    for delta_year in [2,4,6,8]:
        print("delta year: %d" % delta_year)
        parameter_tunning(datetime.date(2017, 8, 1) + datetime.timedelta(days=-365*delta_year), datetime.date(2017, 8, 1))


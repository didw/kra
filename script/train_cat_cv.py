#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import GridSearchCV

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

    print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
    X_data, Y_data, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016_v1.csv')
    #scaler_y = StandardScaler()
    #Y_data = scaler_y.fit_transform(Y_data)
    print("Done")

    for itr in [100,1000]:
        for lr in [10,1]:
            for dep in [3,10]:
                for llr in [1,5]:
                    print("SETUP: iter: %d, lr: %f, depth: %d, l2_leaf_reg: %d" % 
                        (itr, lr, dep, llr))
                    train_pool = Pool(X_data, Y_data, cat_features=cat_feature_inds)
                    scores = []
                    # Use CV
                    params = {'iterations':itr, 
                            'learning_rate':lr,
                            'depth':dep, 
                            'l2_leaf_reg':llr,
                            'loss_function':'MAE',
                            'eval_metric':'MAE'}
                    scores = cv(params, train_pool, fold_count=5)
                    #print(scores)
                    print("train: %.10f, test: %.10f" % (scores['MAE_train_avg'][-1], scores['MAE_test_avg'][-1]))


if __name__ == '__main__':
    for delta_year in [2,3,4,5,6,7,8]:
        print("delta year: %d" % delta_year)
        parameter_tunning(datetime.date(2017, 9, 1) + datetime.timedelta(days=-365*delta_year), datetime.date(2017, 9, 1))


#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import parse_txt_race as pr
import datetime
import pandas as pd
import os.path
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.externals import joblib
import simulation as sim
from mean_data import mean_data
import numpy as np
import time
from etaprogress.progress import ProgressBar
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from multiprocessing import Process, Queue
from catboost import CatBoostRegressor, Pool
from math import sqrt
from sklearn.model_selection import GridSearchCV

MODEL_NUM = 30
NUM_ENSEMBLE = 10

cat_feature_inds = [0,1,2,5]+[i for i in range(12,78)]+[79,80,81,84,85,86]+[i for i in range(124,204)]

def normalize_data(org_data):
    data = org_data.dropna()
    data = data.reset_index()
    return data


def get_data_from_csv(begin_date, end_date, fname_csv, course=0, kind=0, nData=47):
    df = pd.read_csv(fname_csv)
    remove_index = []
    for idx in range(len(df)):
        #print(df['date'][idx])
        date = int(df['date'][idx])
        if date < begin_date or date > end_date or (course > 0 and course != int(df['course'][idx])) or (kind > 0 and kind != int(df['kind'][idx])):
            remove_index.append(idx)
            #print('Delete %dth row (hr: %s, jk: %s, tr: %s, dt: %s)' % (idx, X_data['hr_nt'][idx], X_data['jk_nt'][idx], X_data['tr_nt'][idx], X_data['hr_dt'][idx]))
    data = df.drop(df.index[remove_index])
    data = normalize_data(data)

    R_data = data[['name', 'rank', 'r1', 'r2', 'r3', 'hr_nt', 'hr_dt', 'jk_nt', 'tr_nt', 'cnt', 'rcno', 'price', 'bokyeon1', 'bokyeon2', 'bokyeon3', 'boksik', 'ssang', 'sambok', 'samssang', 'idx']]
    Y_data = data['rctime']
    X_data = data.copy()
    X_data = X_data.drop(['name', 'rctime', 'rank', 'r3', 'r2', 'r1', 'date', 'price', 'bokyeon1', 'bokyeon2', 'bokyeon3', 'boksik', 'ssang', 'sambok', 'ssang', 'samssang', 'index'], axis=1)
    return X_data, Y_data, R_data, data


def training(train_bd, train_ed, q):
    train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
    train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))
    model_dir = "../model/catboost/i500_lr5_dp4_l21/%d_%d" % (train_bd_i, train_ed_i)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
    X_data, Y_data, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016_v1.csv', 0)

    train_pool = Pool(X_data, Y_data, cat_features=cat_feature_inds)

    for i in range(MODEL_NUM):

        model_name = "%s/%d/model.pkl" % (model_dir,i)
        if os.path.exists(model_name):
            print("model[%d] exist. pass.. %s - %s" % (i, str(train_bd), str(train_ed)))
        else:
            if not os.path.exists("%s/%d" % (model_dir, i)):
                os.makedirs("%s/%d" % (model_dir, i))
            print("model[%d] training.." % (i+1))
            train_catboost("%s_%s/%d"%(train_bd_i, train_ed_i, i), train_pool, i)
    print("Finish train model")


def train_catboost(dir_path, X_pool, i):
    model = CatBoostRegressor(
        iterations=500, learning_rate=5.0,
        depth=4, l2_leaf_reg=1,
        thread_count=8,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=i)

    model.fit(X_pool, cat_features=cat_feature_inds)

    if not os.path.exists("../model/catboost/i500_lr5_dp4_l21/%s"%dir_path):
        os.makedirs("../model/catboost/i500_lr5_dp4_l21/%s"%dir_path)
    model.save_model("../model/catboost/i500_lr5_dp4_l21/%s/model.pkl"%dir_path)


def process_train(train_bd, train_ed, q):
    train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
    train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))
    model_dir = "../model/catboost/i500_lr5_dp4_l21/%d_%d" % (train_bd_i, train_ed_i)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
    X_data, Y_data, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016_v1.csv', 0, nData=nData)

    train_pool = Pool(X_data, Y_data, cat_features=cat_feature_inds)

    print("Done")

    for i in range(MODEL_NUM):
        model_name = "%s/%d/model.pkl" % (model_dir,i)
        if os.path.exists(model_name):
            print("model[%d] exist. pass.. %s - %s" % (i, str(train_bd), str(train_ed)))
        else:
            if not os.path.exists("%s/%d" % (model_dir, i)):
                os.makedirs("%s/%d" % (model_dir, i))
            print("model[%d] training.." % (i+1))
            train_catboost("%s_%s/%d"%(train_bd_i, train_ed_i, i), train_pool, i)
    print("Finish train model")


def process_test(train_bd, train_ed, q):
    sr, sscore = q.get()

    test_bd = train_ed + datetime.timedelta(days=1)
    test_ed = train_ed + datetime.timedelta(days=2)
    test_bd_i = int("%d%02d%02d" % (test_bd.year, test_bd.month, test_bd.day))
    test_ed_i = int("%d%02d%02d" % (test_ed.year, test_ed.month, test_ed.day))
    
    train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
    train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))
    data_dir = "../data/catboost/i500_lr5_dp4_l21"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    fname_result = '%s/result.txt' % (data_dir,)
    print("Loading Datadata at %s - %s" % (str(test_bd), str(test_ed)))
    X_test, Y_test, R_test, X_data = get_data_from_csv(test_bd_i, test_ed_i, '../data/1_2007_2016_v1.csv', nData=nData)
    X_pool = Pool(X_test, Y_test, cat_features=cat_feature_inds)
    print("%d data is fully loaded" % (len(X_test)))

    print("train data: %s - %s" % (str(train_bd), str(train_ed)))
    print("test data: %s - %s" % (str(test_bd), str(test_ed)))
    print("%15s%10s%10s%10s%10s%10s%10s%10s" % ("score", "d", "y", "b", "by", "s", "sb", "ss"))

    model = CatBoostRegressor(
        iterations=500, learning_rate=5.0,
        depth=4, l2_leaf_reg=1,
        loss_function='MAE',
        eval_metric='MAE')
    res = [0]*10
    if len(X_test) == 0:
        res = [0]*10
        return
    else:
        DEBUG = False
        Y_test = np.array(Y_test.values.reshape(-1,1)).reshape(-1)
        pred = [0] * MODEL_NUM
        for i in range(MODEL_NUM):
            model.load_model("../model/catboost/i500_lr5_dp4_l21/%d_%d/%d/model.pkl"%(train_bd_i, train_ed_i, i))
            pred[i] = model.predict(X_pool)
            score = np.sqrt(np.mean((pred[i] - Y_test)*(pred[i] - Y_test)))

            res[0] = sim.simulation7(pred[i], R_test, [[1],[2],[3]])
            res[1] = sim.simulation7(pred[i], R_test, [[1,2],[1,2,3],[1,2,3]])
            res[2] = sim.simulation7(pred[i], R_test, [[1,2,3],[1,2,3],[1,2,3]])
            res[3] = sim.simulation7(pred[i], R_test, [[1,2,3,4],[1,2,3,4],[1,2,3,4]])
            res[4] = sim.simulation7(pred[i], R_test, [[3,4,5],[4,5,6],[4,5,6]])
            res[5] = sim.simulation7(pred[i], R_test, [[4,5,6],[4,5,6],[4,5,6,7]])
            res[6] = sim.simulation7(pred[i], R_test, [[4,5,6,7],[4,5,6,7],[4,5,6,7]])

            print("pred[%d] test: " % (i+1), pred[i][:4])
            print("Y_test test: ", Y_test[:4])
            print("result[%02d]: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f" % (
                    i+1, score, res[0], res[1], res[2], res[3], res[4], res[5], res[6]))
            for j in range(7):
                sr[i][j] += res[j]
            sscore[i] += score

            fname_result = '%s/ss_m%02d.txt' % (data_dir, i)
            f_result = open(fname_result, 'a')
            f_result.write("train data: %s - %s\n" % (str(train_bd), str(train_ed)))
            f_result.write("test data: %s - %s\n" % (str(test_bd), str(test_ed)))
            f_result.write("%15s%10s%10s%10s%10s%10s%10s%10s\n" % ("score", "d", "y", "b", "by", "s", "sb", "ss"))
            f_result.write("result: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f\n" % (
                            score, res[0], res[1], res[2], res[3], res[4], res[5], res[6]))
            f_result.close()

        for i in range(7):
            sr[MODEL_NUM][i] = 0
        sscore[MODEL_NUM] = 0

        for i in range(MODEL_NUM):
            for j in range(7):
                sr[MODEL_NUM][j] += 1./MODEL_NUM * sr[i][j]
            sscore[MODEL_NUM] += 1./MODEL_NUM * sscore[i]

        fname_result = '%s/ss_m_all.txt' % data_dir
        f_result = open(fname_result, 'a')
        f_result.write("train data: %s - %s\n" % (str(train_bd), str(train_ed)))
        f_result.write("test data: %s - %s\n" % (str(test_bd), str(test_ed)))
        f_result.write("%15s%10s%10s%10s%10s%10s%10s%10s\n" % ("score", "d", "y", "b", "by", "s", "sb", "ss"))
        f_result.write("result: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f\n" % (
                        sscore[MODEL_NUM], sr[MODEL_NUM][0], sr[MODEL_NUM][1], sr[MODEL_NUM][2], sr[MODEL_NUM][3], sr[MODEL_NUM][4], sr[MODEL_NUM][5], sr[MODEL_NUM][6]))
        f_result.close()

        index_sum = MODEL_NUM + int(MODEL_NUM/NUM_ENSEMBLE) + 1
        for i in range(int(MODEL_NUM/NUM_ENSEMBLE)):
            pred_ens = np.mean(pred[i*NUM_ENSEMBLE:(i+1)*NUM_ENSEMBLE], axis=0)
            score = np.sqrt(np.mean((pred_ens - Y_test)*(pred_ens - Y_test)))

            res[0] = sim.simulation7(pred_ens, R_test, [[1],[2],[3]])
            res[1] = sim.simulation7(pred_ens, R_test, [[1,2],[1,2,3],[1,2,3]])
            res[2] = sim.simulation7(pred_ens, R_test, [[1,2,3],[1,2,3],[1,2,3]])
            res[3] = sim.simulation7(pred_ens, R_test, [[1,2,3,4],[1,2,3,4],[1,2,3,4]])
            res[4] = sim.simulation7(pred_ens, R_test, [[3,4,5],[4,5,6],[4,5,6]])
            res[5] = sim.simulation7(pred_ens, R_test, [[4,5,6],[4,5,6],[4,5,6,7]])
            res[6] = sim.simulation7(pred_ens, R_test, [[4,5,6,7],[4,5,6,7],[4,5,6,7]])

            #print("pred_ens test: ", pred_ens[20:24])
            #print("Y_test test: ", Y_test[20:24])
            print("result_ens[%2d]: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f" % (
                    MODEL_NUM+i+2, score, res[0], res[1], res[2], res[3], res[4], res[5], res[6]))

            index_j = MODEL_NUM+i+1
            for j in range(7):
                sr[index_j][j] += res[j]
            sscore[index_j] += score
            
            fname_result = '%s/ss_ens%d.txt' % (data_dir, i)
            f_result = open(fname_result, 'a')
            f_result.write("train data: %s - %s\n" % (str(train_bd), str(train_ed)))
            f_result.write("test data: %s - %s\n" % (str(test_bd), str(test_ed)))
            f_result.write("%15s%10s%10s%10s%10s%10s%10s%10s\n" % ("score", "d", "y", "b", "by", "s", "sb", "ss"))
            f_result.write("result: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f\n" % (
                            score, res[0], res[1], res[2], res[3], res[4], res[5], res[6]))
            f_result.close()

        sr[index_sum][0] = 0
        sr[index_sum][1] = 0
        sr[index_sum][2] = 0
        sr[index_sum][3] = 0
        sr[index_sum][4] = 0
        sr[index_sum][5] = 0
        sr[index_sum][6] = 0
        sscore[index_sum] = 0
        
        for i in range(int(MODEL_NUM/NUM_ENSEMBLE)):
            index_j = MODEL_NUM+i+1
            sr[index_sum][0] += 1.*NUM_ENSEMBLE/MODEL_NUM * sr[index_j][0]
            sr[index_sum][1] += 1.*NUM_ENSEMBLE/MODEL_NUM * sr[index_j][1]
            sr[index_sum][2] += 1.*NUM_ENSEMBLE/MODEL_NUM * sr[index_j][2]
            sr[index_sum][3] += 1.*NUM_ENSEMBLE/MODEL_NUM * sr[index_j][3]
            sr[index_sum][4] += 1.*NUM_ENSEMBLE/MODEL_NUM * sr[index_j][4]
            sr[index_sum][5] += 1.*NUM_ENSEMBLE/MODEL_NUM * sr[index_j][5]
            sr[index_sum][6] += 1.*NUM_ENSEMBLE/MODEL_NUM * sr[index_j][6]
            sscore[index_sum] += 1.*NUM_ENSEMBLE/MODEL_NUM * sscore[index_j]

        fname_result = '%s/ss_ens_all.txt' % data_dir
        f_result = open(fname_result, 'a')
        f_result.write("train data: %s - %s\n" % (str(train_bd), str(train_ed)))
        f_result.write("test data: %s - %s\n" % (str(test_bd), str(test_ed)))
        f_result.write("%15s%10s%10s%10s%10s%10s%10s%10s\n" % ("score", "d", "y", "b", "by", "s", "sb", "ss"))
        f_result.write("result: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f\n" % (
                        sscore[index_sum], sr[index_sum][0], sr[index_sum][1], sr[index_sum][2], sr[index_sum][3], sr[index_sum][4], sr[index_sum][5], sr[index_sum][6]))
        f_result.close()

    for m in range(MODEL_NUM+int(MODEL_NUM/NUM_ENSEMBLE)+2):
        print("result[%02d]: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f" % (
                m, sscore[m], sr[m][0], sr[m][1], sr[m][2], sr[m][3], sr[m][4], sr[m][5], sr[m][6]))
    q.put((sr, sscore))


def simulation_weekly_train0(begin_date, end_date, delta_day=0, delta_year=0, courses=[0], kinds=[0], nData=47):
    remove_outlier = False
    today = end_date
    sr = [[0 for _ in range(10)] for _ in range(MODEL_NUM+int(MODEL_NUM/NUM_ENSEMBLE)+2)]
    sscore = [0 for _ in range(MODEL_NUM+int(MODEL_NUM/NUM_ENSEMBLE)+2)]
    q = Queue()
    while today >= begin_date:
        while today.weekday() != 4:
            today = today - datetime.timedelta(days=1)
        train_bd = today + datetime.timedelta(days=-365*delta_year)
        #train_bd = datetime.date(2011, 1, 1)
        train_ed = today + datetime.timedelta(days=-delta_day)
        test_bd = today + datetime.timedelta(days=1)
        test_ed = today + datetime.timedelta(days=2)
        today = today - datetime.timedelta(days=2)
        test_bd_s = "%d%02d%02d" % (test_bd.year, test_bd.month, test_bd.day)
        test_ed_s = "%d%02d%02d" % (test_ed.year, test_ed.month, test_ed.day)
        if not os.path.exists('../txt/1/rcresult/rcresult_1_%s.txt' % test_bd_s) and not os.path.exists('../txt/1/rcresult/rcresult_1_%s.txt' % test_ed_s):
            continue
        p = Process(target=process_train, args=(train_bd, train_ed, q))
        p.start()
        p.join()
        q.put((sr, sscore))
        p = Process(target=process_test, args=(train_bd, train_ed, q))
        p.start()
        p.join()
        sr, sscore = q.get()


if __name__ == '__main__':
    delta_year = 4
    train_bd = datetime.date(2011, 11, 1)
    train_ed = datetime.date(2016, 10, 31)
    test_bd = datetime.date(2016, 6, 15)
    test_ed = datetime.date(2017, 9, 30)

    for delta_year in [8]:
        for nData in [186]:
            simulation_weekly_train0(test_bd, test_ed, 0, delta_year, courses=[0], nData=nData)


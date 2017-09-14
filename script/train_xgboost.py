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
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from multiprocessing import Process, Queue
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV

MODEL_NUM = 10
NUM_ENSEMBLE = 5

def dense(input, n_in, n_out, p_keep=0.8):
    weights = tf.get_variable('weight', [n_in, n_out], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', [n_out], initializer=tf.constant_initializer(0.0))
    h1 = tf.matmul(input, weights) + biases
    return tf.nn.elu(tf.nn.batch_normalization(h1, 0.001, 1.0, 0, 1, 0.0001))


def dense_with_onehot(input, n_in, n_out, p_keep=0.8):
    inputs = tf.reshape(tf.one_hot(tf.to_int32(input), depth=n_in, on_value=1.0, off_value=0.0, axis=-1), [-1, n_in])
    weights = tf.get_variable('weight', [n_in, n_out], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', [n_out], initializer=tf.constant_initializer(0.0))
    h1 = tf.matmul(inputs, weights) + biases
    return tf.nn.elu(tf.nn.batch_normalization(h1, 0.001, 1.0, 0, 1, 0.0001))


lg_oh = [5009, 1489, 602, 318, 191, 309, 572, 399, 552, 1373, 769, 396, 717, 1296, 744, 1232, 4450, 1815, 714, 367, 666, 1636, 833, 1505, 4032, 1777, 719, 1604, 3677, 1627, 3320, 18002, 5436, 1570, 619, 336, 592, 1438, 773, 1342, 4712, 1846, 726, 1644, 4187, 1737, 3756, 16473, 5077, 1531, 632, 1399, 4368, 1720, 3858, 14679, 4502, 1394, 3811, 12577, 3783, 10447]
idx_input  = [ 1,  1, 1, 9] + [1]*62 + [ 1,  1, 1, 1, 1,   1,   1,   1,  2,  1,  1,  1, 146]
is_onehot  = [ 1,  1, 1, 0] + [1]*62 + [ 1,  1, 1, 1, 0,   1,   1,   1,  0,  1,  1,  1,   0]
len_onehot = [10, 20, 8, 0] +  lg_oh + [16, 17, 3, 9, 0, 256, 130, 902,  0, 15, 15, 12,   0]
len_h1s    = [ 2,  2, 2, 5] + [2]*62 + [ 2,  2, 1, 2, 1,   2,   2,   2,  2,  2,  2,  2, 100]
name_one_hot_columns = ['course', 'humidity', 'kind', 'idx', 'cntry', 'gender', 'age', 'jockey', 'trainer', 'owner', 'cnt', 'rcno', 'month']

def build_model(input, p_keep):
    inputs = tf.split(input, idx_input, 1)
    h1s = []
    for i in range(len(idx_input)):
        with tf.variable_scope('h1_%d'%i):
            if is_onehot[i] == 1:
                h1s.append(dense_with_onehot(inputs[i], len_onehot[i], len_h1s[i], p_keep))
            else:
                h1s.append(dense(inputs[i], idx_input[i], len_h1s[i], p_keep))
    h1 = tf.concat(h1s, 1)
    """
    with tf.variable_scope('h1'):
        h1 = dense(input, np.sum(idx_input), np.sum(len_h1s))
    """
    with tf.variable_scope('h2'):
        h2 = dense(h1, np.sum(len_h1s), 20, p_keep)

    with tf.variable_scope('h4'):
        weights = tf.get_variable('weight', [np.sum(len_h1s), 1], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0.0))
        #h4 = tf.nn.batch_normalization(h1, 0.001, 1.0, 0, 1, 0.0001)
        return tf.matmul(h1, weights) + biases


def normalize_data(org_data):
    data = org_data.dropna()
    data = data.reset_index()

    column_unique = joblib.load('../data/column_unique.pkl')
    for column in name_one_hot_columns:
        for idx, value in enumerate(column_unique[column]):
            try:
                data.loc[data[column]==value, column] = idx
            except TypeError:
                print(column, idx, value)
                raise
    i = 0
    for row in data.iterrows():
        try:
            int(data.loc[i,'jockey'])
        except ValueError:
            print("jk %s is not exists" % data.loc[i, 'jockey'])
            data.loc[i,'jockey'] = -1
        try:
            int(data.loc[i,'trainer'])
        except ValueError:
            print("tr %s is not exists" % data.loc[i, 'trainer'])
            data.loc[i,'trainer'] = -1
        try:
            int(data.loc[i,'owner'])
        except ValueError:
            print("ow %s is not exists" % data.loc[i, 'owner'])
            data.loc[i,'owner'] = -1
        i += 1
    return data

def get_data(begin_date, end_date):
    train_bd = begin_date
    train_ed = end_date
    date = train_bd
    data = pd.DataFrame()
    first = True
    date += datetime.timedelta(days=-1)
    while date < train_ed:
        date += datetime.timedelta(days=1)
        if date.weekday() != 5 and date.weekday() != 6:
            continue
        filename = "../txt/1/rcresult/rcresult_1_%02d%02d%02d.txt" % (date.year, date.month, date.day)
        if not os.path.isfile(filename):
            continue
        if first:
            data = pr.get_data(filename)
            first = False
        else:
            data = data.append(pr.get_data(filename), ignore_index=True)
    print(data)
    data = normalize_data(data)
    print(data['cnt'])
    print(data['rcno'])
    R_data = data[['rank', 'r1', 'r2', 'r3', 'hr_nt', 'hr_dt', 'jk_nt', 'tr_nt', 'cnt', 'rcno']]
    Y_data = data['rctime']
    X_data = data.copy()
    del X_data['name']
    del X_data['jockey']
    del X_data['trainer']
    del X_data['owner']
    del X_data['rctime']
    del X_data['rank']
    del X_data['r3']
    del X_data['r2']
    del X_data['r1']
    del X_data['date']
    print(R_data)
    return X_data, Y_data, R_data, data


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

def delete_lack_data(X_data, Y_data):
    remove_index = []
    for idx in range(len(X_data)):
        if X_data['hr_nt'][idx] == -1 or X_data['jk_nt'][idx] == -1 or X_data['tr_nt'][idx] == -1:
            remove_index.append(idx)
            #print('Delete %dth row (hr: %s, jk: %s, tr: %s, dt: %s)' % (idx, X_data['hr_nt'][idx], X_data['jk_nt'][idx], X_data['tr_nt'][idx], X_data['hr_dt'][idx]))
    print(len(remove_index))
    return X_data.drop(X_data.index[remove_index]), Y_data.drop(Y_data.index[remove_index])

def training(train_bd, train_ed, q):
    train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
    train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))
    model_dir = "../model/xgboost/ens_e1000/%d_%d" % (train_bd_i, train_ed_i)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
    X_data, Y_data, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016_v1.csv', 0)
    scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_x5, scaler_x6, scaler_y = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
    X_data = np.array(X_data)
    X_data = np.array(X_data)
    X_data, Y_data = shuffle(X_data, Y_data)
    
    X_data[:,3:5] = scaler_x1.fit_transform(X_data[:,3:5])
    X_data[:,6:12] = scaler_x2.fit_transform(X_data[:,6:12])
    X_data[:,78:79] = scaler_x3.fit_transform(X_data[:,78:79])
    X_data[:,82:84] = scaler_x4.fit_transform(X_data[:,82:84])
    X_data[:,88:124] = scaler_x5.fit_transform(X_data[:,88:124])
    X_data[:,204:233] = scaler_x6.fit_transform(X_data[:,204:233])
    Y_data = scaler_y.fit_transform(Y_data.reshape(-1,1))
    print("Done")

    joblib.dump((scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_x5, scaler_x6), "%s/scaler_x.pkl" % (model_dir,))
    joblib.dump(scaler_y, "%s/scaler_y.pkl" % (model_dir,))

    for i in range(MODEL_NUM):
        if i < 5:
            X_train, X_val, y_train, y_val = train_test_split(X_data, Y_data, random_state=np.random.randint(100))
        else:
            X_train, y_train = X_data, Y_data
            X_val, y_val = None, None

        model_name = "%s/%d/model.pkl" % (model_dir,i)
        if os.path.exists(model_name):
            print("model[%d] exist. pass.. %s - %s" % (i, str(train_bd), str(train_ed)))
        else:
            if not os.path.exists("%s/%d" % (model_dir, i)):
                os.makedirs("%s/%d" % (model_dir, i))
            print("model[%d] training.." % (i+1))
            train_xgboost("%s_%s/%d"%(train_bd_i, train_ed_i, i), X_train, y_train, X_val, y_val)
    print("Finish train model")
    q.put(scaler_x1)
    q.put(scaler_x2)
    q.put(scaler_x3)
    q.put(scaler_x4)
    q.put(scaler_x5)
    q.put(scaler_x6)
    q.put(scaler_y)


def print_log(data, pred, fname):
    flog = open(fname, 'w')
    rcno = 1
    flog.write("rcno\tcourse\tidx\tname\tcntry\tgender\tage\tbudam\tjockey\ttrainer\tweight\tdweight\thr_days\thumidity\thr_nt\thr_nt1\thr_nt2\thr_ny\thr_ny1\thr_ny2\t")
    flog.write("jk_nt\tjk_nt1\tjk_nt2\tjk_ny\tjk_ny1\tjk_ny2\ttr_nt\ttr_nt1\ttr_nt2\ttr_ny\ttr_ny1\ttr_ny2\tpredict\n")
    for idx in range(len(data)):
        if rcno != data['rcno'][idx]:
            rcno = data['rcno'][idx]
            flog.write('\n')
        flog.write("%s\t%s\t%s\t%s\t%s\t" % (data['rcno'][idx], data['course'][idx], data['idx'][idx], data['name'][idx], data['cntry'][idx]))
        flog.write("%s\t%s\t%s\t%s\t%s\t" % (data['gender'][idx], data['age'][idx], data['budam'][idx], data['jockey'][idx], data['trainer'][idx]))
        flog.write("%s\t%s\t%s\t%s\t" % (data['weight'][idx], data['dweight'][idx], data['hr_days'][idx], data['humidity'][idx]))
        flog.write("%s\t%s\t%s\t%s\t%s\t%s\t" % (data['hr_nt'][idx], data['hr_nt1'][idx], data['hr_nt2'][idx], data['hr_ny'][idx], data['hr_ny1'][idx], data['hr_ny2'][idx]))
        flog.write("%s\t%s\t%s\t%s\t%s\t%s\t" % (data['jk_nt'][idx], data['jk_nt1'][idx], data['jk_nt2'][idx], data['jk_ny'][idx], data['jk_ny1'][idx], data['jk_ny2'][idx]))
        flog.write("%s\t%s\t%s\t%s\t%s\t%s\t" % (data['tr_nt'][idx], data['tr_nt1'][idx], data['tr_nt2'][idx], data['tr_ny'][idx], data['tr_ny1'][idx], data['tr_ny2'][idx]))
        flog.write("%f\n" % pred['predict'][idx])
    flog.close()


def train_xgboost(dir_path, X_train, y_train, X_val, y_val):
    max_depth = 3
    min_child_weight = 10
    subsample = 0.5
    colsample_bytree = 0.6
    objective = 'reg:linear'
    num_estimators = 100
    learning_rate = 1e-1
    n_jobs = 10
    xg_train = xgb.DMatrix(X_train, label=y_train)
    if y_val is not None:
        xg_val = xgb.DMatrix(X_val, label=y_val)
    # setup parameters for xgboost
    param = {}
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 12
    param['num_class'] = 1
    if y_val is not None:
        watchlist = [(xg_train, 'train'), (xg_val, 'val')]
    else:
        watchlist = [(xg_train, 'train')]
    # use linear regression
    param['objective'] = 'reg:linear'
    num_round = 1000
    bst = xgb.train(param, xg_train, num_round, watchlist, verbose_eval=False)
    # get prediction
    if y_val is not None:
        pred = bst.predict(X_val)
        error_rate = sqrt(mean_squared_error(pred, y_val))
        print('Test error using softmax = {}'.format(error_rate))
    if not os.path.exists("../model/xgboost/ens_e1000/%s"%dir_path):
        os.makedirs("../model/xgboost/ens_e1000/%s"%dir_path)
    joblib.dump(bst, "../model/xgboost/ens_e1000/%s/model.pkl"%dir_path)


def process_train(train_bd, train_ed, q):
    train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
    train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))
    model_dir = "../model/xgboost/ens_e1000/%d_%d" % (train_bd_i, train_ed_i)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
    X_data, Y_data, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016_v1.csv', 0, nData=nData)
    scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_x5, scaler_x6, scaler_y = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
    X_data = np.array(X_data)
    X_data = np.array(X_data)
    X_data, Y_data = shuffle(X_data, Y_data)
    
    X_data[:,3:5] = scaler_x1.fit_transform(X_data[:,3:5])
    X_data[:,6:12] = scaler_x2.fit_transform(X_data[:,6:12])
    X_data[:,78:79] = scaler_x3.fit_transform(X_data[:,78:79])
    X_data[:,82:84] = scaler_x4.fit_transform(X_data[:,82:84])
    X_data[:,88:124] = scaler_x5.fit_transform(X_data[:,88:124])
    X_data[:,204:233] = scaler_x6.fit_transform(X_data[:,204:233])
    Y_data = scaler_y.fit_transform(Y_data.reshape(-1,1))
    print("Done")

    joblib.dump((scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_x5, scaler_x6), "%s/scaler_x.pkl" % (model_dir,))
    joblib.dump(scaler_y, "%s/scaler_y.pkl" % (model_dir,))

    for i in range(MODEL_NUM):
        if i < 5:
            X_train, X_val, y_train, y_val = train_test_split(X_data, Y_data, random_state=np.random.randint(100))
        else:
            X_train, y_train = X_data, Y_data
            X_val, y_val = None, None

        model_name = "%s/%d/model.pkl" % (model_dir,i)
        if os.path.exists(model_name):
            print("model[%d] exist. pass.. %s - %s" % (i, str(train_bd), str(train_ed)))
        else:
            if not os.path.exists("%s/%d" % (model_dir, i)):
                os.makedirs("%s/%d" % (model_dir, i))
            print("model[%d] training.." % (i+1))
            train_xgboost("%s_%s/%d"%(train_bd_i, train_ed_i, i), X_train, y_train, X_val, y_val)
    print("Finish train model")
    q.put(scaler_x1)
    q.put(scaler_x2)
    q.put(scaler_x3)
    q.put(scaler_x4)
    q.put(scaler_x5)
    q.put(scaler_x6)
    q.put(scaler_y)


def load_g_estimators():
    g_estimators = [0] * MODEL_NUM
    for i in range(MODEL_NUM):
        print("load g_estimators[%d]..." % i)
        g_estimators[i] = TensorflowRegressor('reference/%d' % i)
        #g_estimators[i].load()
    return g_estimators


def process_test(train_bd, train_ed, scaler, q):
    scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_x5, scaler_x6, scaler_y = scaler
    sr, sscore = q.get()

    test_bd = train_ed + datetime.timedelta(days=1)
    test_ed = train_ed + datetime.timedelta(days=2)
    test_bd_i = int("%d%02d%02d" % (test_bd.year, test_bd.month, test_bd.day))
    test_ed_i = int("%d%02d%02d" % (test_ed.year, test_ed.month, test_ed.day))
    
    train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
    train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))
    data_dir = "../data/xgboost/ens_e1000_20100814_20160812"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    fname_result = '%s/result.txt' % (data_dir,)
    print("Loading Datadata at %s - %s" % (str(test_bd), str(test_ed)))
    X_test, Y_test, R_test, X_data = get_data_from_csv(test_bd_i, test_ed_i, '../data/1_2007_2016_v1.csv', nData=nData)
    X_test = np.array(X_test)
    X_test[:,3:5] = scaler_x1.transform(X_test[:,3:5])
    X_test[:,6:12] = scaler_x2.transform(X_test[:,6:12])
    X_test[:,78:79] = scaler_x3.transform(X_test[:,78:79])
    X_test[:,82:84] = scaler_x4.transform(X_test[:,82:84])
    X_test[:,88:124] = scaler_x5.transform(X_test[:,88:124])
    X_test[:,204:233] = scaler_x6.transform(X_test[:,204:233])
    """
    X_test = scaler_x1.transform(X_test)
    """
    print("%d data is fully loaded" % (len(X_test)))

    print("train data: %s - %s" % (str(train_bd), str(train_ed)))
    print("test data: %s - %s" % (str(test_bd), str(test_ed)))
    print("%15s%10s%10s%10s%10s%10s%10s%10s" % ("score", "d", "y", "b", "by", "s", "sb", "ss"))

    res = [0]*10
    if len(X_test) == 0:
        res = [0]*10
        return
    else:
        DEBUG = False
        if DEBUG:
            X_test.to_csv('../log/weekly_train0_%s.csv' % today, index=False)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test.values.reshape(-1,1)).reshape(-1)
        pred = [0] * MODEL_NUM
        for i in range(MODEL_NUM):
            #estimator = joblib.load("../model/xgboost/ens_e1000/%d_%d/%d/model.pkl"%(train_bd_i, train_ed_i, i))
            estimator = joblib.load("../model/xgboost/ens_e1000/20100814_20160812/%d/model.pkl"%i)
            xg_test = xgb.DMatrix(X_test)
            pred[i] = estimator.predict(X_test)
            pred[i] = scaler_y.inverse_transform(pred[i])
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
            n_split = int(MODEL_NUM/NUM_ENSEMBLE)
            pred_ens = np.mean(pred[i*n_split:(i+1)*n_split], axis=0)
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
        #p = Process(target=process_train, args=(train_bd, train_ed, q))
        #p.start()
        #p.join()
        #scaler_x1 = q.get()
        #scaler_x2 = q.get()
        #scaler_x3 = q.get()
        #scaler_x4 = q.get()
        #scaler_x5 = q.get()
        #scaler_x6 = q.get()
        #scaler_y = q.get()
        scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_x5, scaler_x6 = joblib.load('../model/xgboost/ens_e1000/20100814_20160812/scaler_x.pkl')
        scaler_y = joblib.load('../model/xgboost/ens_e1000/20100814_20160812/scaler_y.pkl')
        q.put((sr, sscore))
        p = Process(target=process_test, args=(train_bd, train_ed, (scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_x5, scaler_x6, scaler_y), q))
        p.start()
        p.join()
        sr, sscore = q.get()


if __name__ == '__main__':
    delta_year = 4
    train_bd = datetime.date(2011, 11, 1)
    train_ed = datetime.date(2016, 10, 31)
    test_bd = datetime.date(2016, 6, 5)
    test_ed = datetime.date(2017, 9, 15)

    for delta_year in [6]:
        for nData in [186]:
            simulation_weekly_train0(test_bd, test_ed, 0, delta_year, courses=[0], nData=nData)


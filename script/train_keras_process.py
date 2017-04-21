#!/usr/bin/python
# -*- coding: utf-8 -*-

import parse_txt_race as pr
import datetime
import pandas as pd
import os.path
from sklearn.externals import joblib
import random
import simulation as sim
from mean_data import mean_data
import numpy as np
import pickle
import tensorflow as tf
import time
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process, Queue

MODEL_NUM = 30
NUM_ENSEMBLE = 6


def normalize_data(org_data):
    data = org_data.dropna()
    data = data.reset_index()
    data.loc[data['gender'] == '암', 'gender'] = 0
    data.loc[data['gender'] == '수', 'gender'] = 1
    data.loc[data['gender'] == '거', 'gender'] = 2
    data.loc[data['cntry'] == '한', 'cntry'] = 0
    data.loc[data['cntry'] == '한(포)', 'cntry'] = 1
    data.loc[data['cntry'] == '일', 'cntry'] = 2
    data.loc[data['cntry'] == '중', 'cntry'] = 3
    data.loc[data['cntry'] == '미', 'cntry'] = 4
    data.loc[data['cntry'] == '캐', 'cntry'] = 5
    data.loc[data['cntry'] == '뉴', 'cntry'] = 6
    data.loc[data['cntry'] == '호', 'cntry'] = 7
    data.loc[data['cntry'] == '브', 'cntry'] = 8
    data.loc[data['cntry'] == '헨', 'cntry'] = 9
    data.loc[data['cntry'] == '남', 'cntry'] = 10
    data.loc[data['cntry'] == '아일', 'cntry'] = 11
    data.loc[data['cntry'] == '모', 'cntry'] = 12
    data.loc[data['cntry'] == '영', 'cntry'] = 13
    data.loc[data['cntry'] == '인', 'cntry'] = 14
    data.loc[data['cntry'] == '아', 'cntry'] = 15
    data.loc[data['cntry'] == '프', 'cntry'] = 16
    oh_course = [[0]*13 for _ in range(len(data))]
    oh_gen = [[0]*3 for _ in range(len(data))]
    oh_cnt = [[0]*17 for _ in range(len(data))]
    course_list = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2300]
    for i in range(len(data)):
        oh_course[i][course_list.index(data['course'][i])] = 1
        oh_gen[i][data['gender'][i]] = 1
        oh_cnt[i][data['cntry'][i]] = 1
    df_course = pd.DataFrame(oh_course, columns=['cr%d'%i for i in range(1,14)])
    df_gen = pd.DataFrame(oh_gen, columns=['g1', 'g2', 'g3'])
    df_cnt = pd.DataFrame(oh_cnt, columns=['c%d'%i for i in range(1,18)])
    return pd.concat([data, df_course, df_gen, df_cnt], axis=1)
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


def get_data_from_csv(begin_date, end_date, fname_csv, course=0, kind=0, nData=201):
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
    X_data = X_data.drop(['name', 'jockey', 'trainer', 'owner', 'rctime', 'rank', 'r3', 'r2', 'r1', 'date', 'price', 'bokyeon1', 'bokyeon2', 'bokyeon3', 'boksik', 'ssang', 'sambok', 'ssang', 'samssang', 'index'], axis=1)
    if nData == 11:
        X_data = X_data.drop(['humidity', 'kind', 'dbudam', 'drweight', 'lastday', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'ts6', # 12
                  'idx', 'cntry', 'gender', 'age', 'budam', # 9
                  'weight', 'dweight', 'cnt', 'rcno', 'month',
                  'hr_days', 'hr_nt', 'hr_nt1', 'hr_nt2', 'hr_t1', 'hr_t2', 'hr_ny', 'hr_ny1', 'hr_ny2', 'hr_y1', 'hr_y2', # 11
                  'hr_dt', 'hr_d1', 'hr_d2', 'hr_rh', 'hr_rm', 'hr_rl', # 6
                  'jk_nt', 'jk_nt1', 'jk_nt2', 'jk_t1', 'jk_t2', 'jk_ny', 'jk_ny1', 'jk_ny2', 'jk_y1', 'jk_y2', # 10
                  'tr_nt', 'tr_nt1', 'tr_nt2', 'tr_t1', 'tr_t2', 'tr_ny', 'tr_ny1', 'tr_ny2', 'tr_y1', 'tr_y2',  #10
                  'rd1', 'rd2', 'rd3', 'rd4', 'rd5', 'rd6', 'rd7', 'rd8', 'rd9', 'rd10', 'rd11', 'rd12', 'rd13', 'rd14', 'rd15', 'rd16', 'rd17', 'rd18', # 18
                  'jc1', 'jc2', 'jc3', 'jc4', 'jc5', 'jc6', 'jc7', 'jc8', 'jc9', 'jc10', 'jc11', 'jc12', 'jc13', 'jc14', 'jc15', 'jc16', 'jc17', 'jc18', 'jc19', 'jc20', 'jc21', 'jc22', 'jc23', 'jc24', 'jc25', 'jc26', 'jc27', 'jc28', 'jc29', 'jc30',  # 30
                  'jc31', 'jc32', 'jc33', 'jc34', 'jc35', 'jc36', 'jc37', 'jc38', 'jc39', 'jc40', 'jc41', 'jc42', 'jc43', 'jc44', 'jc45', 'jc46', 'jc47', 'jc48', 'jc49', 'jc50', 'jc51', 'jc52', 'jc53', 'jc54', 'jc55', 'jc56', 'jc57', 'jc58', 'jc59', 'jc60',  # 30
                  'jc61', 'jc62', 'jc63', 'jc64', 'jc65', 'jc66', 'jc67', 'jc68', 'jc69', 'jc70', 'jc71', 'jc72', 'jc73', 'jc74', 'jc75', 'jc76', 'jc77', 'jc78', 'jc79', 'jc80', 'jc81',  # 21
                  ], axis=1)
    if nData == 29:
        X_data = X_data.drop(['humidity', 'kind', 'dbudam', 'drweight', 'lastday', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'ts6', # 12
                  'idx', 'cntry', 'gender', 'age', 'budam', # 9
                  'weight', 'dweight', 'cnt', 'rcno', 'month',
                  'hr_days', 'hr_nt', 'hr_nt1', 'hr_nt2', 'hr_t1', 'hr_t2', 'hr_ny', 'hr_ny1', 'hr_ny2', 'hr_y1', 'hr_y2', # 11
                  'hr_dt', 'hr_d1', 'hr_d2', 'hr_rh', 'hr_rm', 'hr_rl', # 6
                  'jk_nt', 'jk_nt1', 'jk_nt2', 'jk_t1', 'jk_t2', 'jk_ny', 'jk_ny1', 'jk_ny2', 'jk_y1', 'jk_y2', # 10
                  'tr_nt', 'tr_nt1', 'tr_nt2', 'tr_t1', 'tr_t2', 'tr_ny', 'tr_ny1', 'tr_ny2', 'tr_y1', 'tr_y2',  #10
                  'jc1', 'jc2', 'jc3', 'jc4', 'jc5', 'jc6', 'jc7', 'jc8', 'jc9', 'jc10', 'jc11', 'jc12', 'jc13', 'jc14', 'jc15', 'jc16', 'jc17', 'jc18', 'jc19', 'jc20', 'jc21', 'jc22', 'jc23', 'jc24', 'jc25', 'jc26', 'jc27', 'jc28', 'jc29', 'jc30',  # 30
                  'jc31', 'jc32', 'jc33', 'jc34', 'jc35', 'jc36', 'jc37', 'jc38', 'jc39', 'jc40', 'jc41', 'jc42', 'jc43', 'jc44', 'jc45', 'jc46', 'jc47', 'jc48', 'jc49', 'jc50', 'jc51', 'jc52', 'jc53', 'jc54', 'jc55', 'jc56', 'jc57', 'jc58', 'jc59', 'jc60',  # 30
                  'jc61', 'jc62', 'jc63', 'jc64', 'jc65', 'jc66', 'jc67', 'jc68', 'jc69', 'jc70', 'jc71', 'jc72', 'jc73', 'jc74', 'jc75', 'jc76', 'jc77', 'jc78', 'jc79', 'jc80', 'jc81',  # 21
                  ], axis=1)
    if nData == 118:
        X_data = X_data.drop(['kind', 'dbudam', 'drweight', 'lastday', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'ts6', # 12
                  'weight', 'dweight', 'rcno',
                  'hr_days', 'hr_nt', 'hr_nt1', 'hr_nt2', 'hr_t1', 'hr_t2', 'hr_ny', 'hr_ny1', 'hr_ny2', 'hr_y1', 'hr_y2', # 11
                  'hr_dt', 'hr_d1', 'hr_d2', 'hr_rh', 'hr_rm', 'hr_rl', # 6
                  'jk_nt', 'jk_nt1', 'jk_nt2', 'jk_t1', 'jk_t2', 'jk_ny', 'jk_ny1', 'jk_ny2', 'jk_y1', 'jk_y2', # 10
                  'tr_nt', 'tr_nt1', 'tr_nt2', 'tr_t1', 'tr_t2', 'tr_ny', 'tr_ny1', 'tr_ny2', 'tr_y1', 'tr_y2',  #10
                  ], axis=1)
    if nData == 47:
        X_data = X_data.drop(['ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'ts6', 'score1', 'score2', 'score3', 'score4', 'score5', 'score6', 'score7', 'score8', 'score9', 'score10', 'hr_dt', 'hr_d1', 'hr_d2', 'hr_rh', 'hr_rm', 'hr_rl'], axis=1)
        X_data = X_data.drop(['rd1', 'rd2', 'rd3', 'rd4', 'rd5', 'rd6', 'rd7', 'rd8', 'rd9', 'rd10', 'rd11', 'rd12', 'rd13', 'rd14', 'rd15', 'rd16', 'rd17', 'rd18', # 18
                  'jc1', 'jc2', 'jc3', 'jc4', 'jc5', 'jc6', 'jc7', 'jc8', 'jc9', 'jc10', 'jc11', 'jc12', 'jc13', 'jc14', 'jc15', 'jc16', 'jc17', 'jc18', 'jc19', 'jc20', 'jc21', 'jc22', 'jc23', 'jc24', 'jc25', 'jc26', 'jc27', 'jc28', 'jc29', 'jc30',
                  'jc31', 'jc32', 'jc33', 'jc34', 'jc35', 'jc36', 'jc37', 'jc38', 'jc39', 'jc40', 'jc41', 'jc42', 'jc43', 'jc44', 'jc45', 'jc46', 'jc47', 'jc48', 'jc49', 'jc50', 'jc51', 'jc52', 'jc53', 'jc54', 'jc55', 'jc56', 'jc57', 'jc58', 'jc59', 'jc60',
                  'jc61', 'jc62', 'jc63', 'jc64', 'jc65', 'jc66', 'jc67', 'jc68', 'jc69', 'jc70', 'jc71', 'jc72', 'jc73', 'jc74', 'jc75', 'jc76', 'jc77', 'jc78', 'jc79', 'jc80', 'jc81'], axis=1)
    if nData == 105:
        X_data = X_data.drop(['jc1', 'jc2', 'jc3', 'jc4', 'jc5', 'jc6', 'jc7', 'jc8', 'jc9', 'jc10', 'jc11', 'jc12', 'jc13', 'jc14', 'jc15', 'jc16', 'jc17', 'jc18', 'jc19', 'jc20', 'jc21', 'jc22', 'jc23', 'jc24', 'jc25', 'jc26', 'jc27', 'jc28', 'jc29', 'jc30',
                  'jc31', 'jc32', 'jc33', 'jc34', 'jc35', 'jc36', 'jc37', 'jc38', 'jc39', 'jc40', 'jc41', 'jc42', 'jc43', 'jc44', 'jc45', 'jc46', 'jc47', 'jc48', 'jc49', 'jc50', 'jc51', 'jc52', 'jc53', 'jc54', 'jc55', 'jc56', 'jc57', 'jc58', 'jc59', 'jc60',
                  'jc61', 'jc62', 'jc63', 'jc64', 'jc65', 'jc66', 'jc67', 'jc68', 'jc69', 'jc70', 'jc71', 'jc72', 'jc73', 'jc74', 'jc75', 'jc76', 'jc77', 'jc78', 'jc79', 'jc80', 'jc81'], axis=1)
    return np.array(X_data), np.array(Y_data), R_data, data

def delete_lack_data(X_data, Y_data):
    remove_index = []
    for idx in range(len(X_data)):
        if X_data['hr_nt'][idx] == -1 or X_data['jk_nt'][idx] == -1 or X_data['tr_nt'][idx] == -1:
            remove_index.append(idx)
            #print('Delete %dth row (hr: %s, jk: %s, tr: %s, dt: %s)' % (idx, X_data['hr_nt'][idx], X_data['jk_nt'][idx], X_data['tr_nt'][idx], X_data['hr_dt'][idx]))
    print(len(remove_index))
    return X_data.drop(X_data.index[remove_index]), Y_data.drop(Y_data.index[remove_index])

def training(train_bd, train_ed, course=0, nData=47):
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.models import model_from_json
    train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
    train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))

    #os.system('rm -r \"../model/keras/e200_i151/%d_%d/\"' % (train_bd_i, train_ed_i))
    model_dir = '../model/keras/e200_i151/%d_%d/' % (train_bd_i, train_ed_i)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = "%s/model_v1.h5" % model_dir
    estimators = [0] * MODEL_NUM
    print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
    X_train, Y_train, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016_v1.csv')
    print("%d data is fully loaded" % len(X_train))
    #X_scaler = StandardScaler()
    #X_train = X_scaler.fit_transform(X_train)
    print("Start train model")
    # fix random seed for reproducibility
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    Y_train = scaler_y.fit_transform(Y_train)
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    for i in range(MODEL_NUM):
        if os.path.exists(model_name.replace('h5', '%d.h5'%i)):
            from keras.models import model_from_json
            print("model[%d] exist. try to loading.. %s - %s" % (i, str(train_bd), str(train_ed)))
            estimators[i] = model_from_json(open(model_name.replace('h5', 'json')).read())
            estimators[i].load_weights(model_name.replace('h5', '%d.h5'%i))
        else:
            print("model[%d] training.." % (i+1))
            def baseline_model():
                from keras.models import Sequential
                from keras.layers import Dense, Dropout
                # create model
                model = Sequential()
                model.add(Dense(128, input_shape=(201,), kernel_initializer='he_normal', activation='relu'))
                model.add(Dense(1, kernel_initializer='he_normal'))
                # Compile model
                model.compile(loss='mean_squared_error', optimizer='adam')
                return model

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            from keras import backend as K
            K.set_session(sess)
            from keras.wrappers.scikit_learn import KerasRegressor
            estimators[i] = KerasRegressor(build_fn=baseline_model, nb_epoch=200, batch_size=32, verbose=0)
            estimators[i].fit(X_train, Y_train)
            # saving model
            json_model = estimators[i].model.to_json()
            open(model_name.replace('h5', 'json'), 'w').write(json_model)
            estimators[i].model.save_weights(model_name.replace('h5', '%d.h5'%i), overwrite=True)
    md = joblib.load('../data/1_2007_2016_v1_md.pkl')
    return estimators, md, [scaler_x, scaler_y]

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


def process_train(train_bd, train_ed):
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    
    train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
    train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))

    model_dir = "../model/keras/e200_i151/%d_%d" % (train_bd_i, train_ed_i)
    model_name = "%s/model_v1.h5" % (model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    estimators = [0] * MODEL_NUM
    print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
    X_train, Y_train, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016_v1.csv', 0, nData=nData)
    #scaler_x = StandardScaler()
    #scaler_y = StandardScaler()
    #X_train = scaler_x.fit_transform(X_train)
    #Y_train = scaler_y.fit_transform(Y_train)
    #joblib.dump(scaler_x, '%s/scaler_x.pkl'%model_dir)
    #joblib.dump(scaler_y, '%s/scaler_y.pkl'%model_dir)
    for i in range(MODEL_NUM):
        if os.path.exists(model_name.replace('h5', '%d.h5'%i)):
            from keras.models import model_from_json
            print("model[%d] exist. try to loading.. %s - %s" % (i, str(train_bd), str(train_ed)))
            estimators[i] = model_from_json(open(model_name.replace('h5', 'json')).read())
            estimators[i].load_weights(model_name.replace('h5', '%d.h5'%i))
        else:
            print("model[%d] training.." % (i+1))
            def baseline_model():
                from keras.models import Sequential
                from keras.layers import Dense, Dropout
                from keras import regularizers
                # create model
                model = Sequential()
                model.add(Dense(150, input_shape=(151,), kernel_initializer='he_normal', activation='relu'))
                model.add(Dense(1, kernel_initializer='he_normal'))
                # Compile model
                model.compile(loss='mean_squared_error', optimizer='adam')
                return model

            from keras.wrappers.scikit_learn import KerasRegressor
            estimators[i] = KerasRegressor(build_fn=baseline_model, nb_epoch=200, batch_size=32, verbose=0)
            estimators[i].fit(X_train, Y_train, epochs=10)
            # saving model
            json_model = estimators[i].model.to_json()
            open(model_name.replace('h5', 'json'), 'w').write(json_model)
            estimators[i].model.save_weights(model_name.replace('h5', '%d.h5'%i), overwrite=True)
    print("Finish train model")


def process_test(train_bd, train_ed, q):
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    sr, sscore = q.get()

    test_bd = train_ed + datetime.timedelta(days=1)
    test_ed = train_ed + datetime.timedelta(days=2)
    test_bd_i = int("%d%02d%02d" % (test_bd.year, test_bd.month, test_bd.day))
    test_ed_i = int("%d%02d%02d" % (test_ed.year, test_ed.month, test_ed.day))
    
    train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
    train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))
    model_dir = "../model/keras/e200_i151/%d_%d" % (train_bd_i, train_ed_i)
    model_name = "%s/model_v1.h5" % (model_dir)
    data_dir = "../data/keras/e200_i151"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print("Loading Datadata at %s - %s" % (str(test_bd), str(test_ed)))

    X_test, Y_test, R_test, _ = get_data_from_csv(test_bd_i, test_ed_i, '../data/1_2007_2016_v1.csv', nData=nData)
    #scaler_x = joblib.load('%s/scaler_x.pkl'%model_dir)
    #scaler_y = joblib.load('%s/scaler_y.pkl'%model_dir)
    #X_test = scaler_x.transform(X_test)
    print("%d data is fully loaded" % (len(X_test)))

    #print("train data: %s - %s" % (str(train_bd), str(train_ed)))
    #print("test data: %s - %s" % (str(test_bd), str(test_ed)))
    print("%15s%10s%10s%10s%10s%10s%10s%10s" % ("score", "d", "y", "b", "by", "s", "sb", "ss"))

    res = [0]*10
    if len(X_test) == 0:
        res = [0]*10
        return
    else:
        DEBUG = False
        if DEBUG:
            X_test.to_csv('../log/weekly_train0_%s.csv' % today, index=False)
        pred = [0] * MODEL_NUM
        for i in range(MODEL_NUM):
            from keras.models import model_from_json
            estimator = model_from_json(open(model_name.replace('h5', 'json')).read())
            estimator.load_weights(model_name.replace('h5', '%d.h5'%i))
            pred[i] = estimator.predict(X_test).flatten()
            #pred[i] = scaler_y.inverse_transform(pred[i])
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

        """
        res[0] = sim.simulation1(pred, R_test, 1)
        res[1] = sim.simulation2(pred, R_test, 1)
        res[2] = sim.simulation3(pred, R_test, [[1,2]])
        res[3] = sim.simulation4(pred, R_test, [1,2])
        res[4] = sim.simulation5(pred, R_test, [[1,2]])
        res[5] = sim.simulation6(pred, R_test, [[1,2,3]])
        res[6] = sim.simulation7(pred, R_test, [[1,2,3],[1,2,3],[2,3,4]])

        res[0] = sim.simulation5(pred, R_test, [[1,2]])
        res[1] = sim.simulation5(pred, R_test, [[1,2],[1,3]])
        res[2] = sim.simulation5(pred, R_test, [[1,2],[1,3],[2,3]])
        res[3] = sim.simulation5(pred, R_test, [[1,2],[1,3],[2,3],[1,4]])
        res[4] = sim.simulation5(pred, R_test, [[1,2],[1,3],[2,3],[1,4],[2,4],[3,4]])
        res[5] = sim.simulation5(pred, R_test, [[1,2],[1,3],[2,3],[1,4],[2,4],[3,4],[1,5]])
        res[6] = sim.simulation5(pred, R_test, [[1,2],[1,3],[2,3],[1,4],[2,4],[3,4],[1,5],[2,5],[3,5],[4,5]])
        
        res[0] = sim.simulation6(pred, R_test, [[1,2,3]])
        res[1] = sim.simulation6(pred, R_test, [[1,2,3], [1,2,4], [1,3,4], [2,3,4]])
        res[2] = sim.simulation6(pred, R_test, [[1,2,3], [1,2,4], [1,2,5], [1,3,4], [1,3,5], [1,4,5], [2,3,4], [2,3,5], [2,4,5], [3,4,5]])
        res[3] = sim.simulation6(pred, R_test, [[1,2,3], [1,2,4], [1,2,5], [1,3,4], [1,3,5], [1,4,5], [2,3,4], [2,3,5], [2,4,5], [3,4,5],
                                            [1,2,6], [1,3,6], [1,4,6], [1,5,6], [2,3,6], [2,4,6], [2,5,6], [3,4,6], [3,5,6], [4,5,6]])
        res[4] = sim.simulation6(pred, R_test, [[1,2,3], [1,2,4], [1,2,5], [1,3,4], [1,3,5], [1,4,5], [2,3,4], [2,3,5], [2,4,5], [3,4,5],
                                            [1,2,6], [1,3,6], [1,4,6], [1,5,6], [2,3,6], [2,4,6], [2,5,6], [3,4,6], [3,5,6], [4,5,6],
                                            [1,2,7], [1,3,7], [1,4,7], [1,5,7], [2,3,7], [2,4,7], [2,5,7], [3,4,7], [3,5,7], [4,5,7],
                                            [1,6,7], [2,6,7], [3,6,7], [4,6,7], [5,6,7]
                                            ])
        res[5] = sim.simulation6(pred, R_test, [[2,3,4], [2,3,5], [2,4,5], [3,4,5], [2,3,6], [2,4,6], [2,5,6], [3,4,6], [3,5,6], [4,5,6]])
        res[6] = sim.simulation6(pred, R_test, [[3,4,5], [3,4,6], [3,4,7], [3,5,6], [3,5,7], [3,6,7], [4,5,6], [4,5,7], [4,6,7], [5,6,7]])
        
        res[0] = sim.simulation7(pred, R_test, [[1],[2],[3]])
        res[1] = sim.simulation7(pred, R_test, [[1,2],[1,2,3],[1,2,3]])
        res[2] = sim.simulation7(pred, R_test, [[1,2,3],[1,2,3,4,5],[1,2,3,4,5,6]])
        res[3] = sim.simulation7(pred, R_test, [[1,2,3,4],[1,2,3,4,5,6],[3,4,5,6]])
        res[4] = sim.simulation7(pred, R_test, [[4,5,6],[4,5,6],[4,5,6]])
        res[5] = sim.simulation7(pred, R_test, [[4,5,6,7,8],[4,5,6,7,8],[4,5,6,7,8]])
        res[6] = sim.simulation7(pred, R_test, [[5,6,7,8,9,10],[5,6,7,8,9,10],[5,6,7,8,9,10]])
        
        res[0] = sim.simulation2(pred, R_test, 1)
        res[1] = sim.simulation2(pred, R_test, 2)
        res[2] = sim.simulation2(pred, R_test, 3)
        res[3] = sim.simulation2(pred, R_test, 4)
        res[4] = sim.simulation2(pred, R_test, 5)
        res[5] = sim.simulation2(pred, R_test, 6)
        res[6] = sim.simulation2(pred, R_test, 7)
        """
    for m in range(MODEL_NUM+int(MODEL_NUM/NUM_ENSEMBLE)+2):
        print("result[%02d]: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f" % (
                m, sscore[m], sr[m][0], sr[m][1], sr[m][2], sr[m][3], sr[m][4], sr[m][5], sr[m][6]))
    q.put((sr, sscore))



def simulation_weekly_train0(begin_date, end_date, delta_day=0, delta_year=0, courses=[0], kinds=[0], nData=47):
    remove_outlier = False
    today = begin_date
    sr = [[0 for _ in range(10)] for _ in range(MODEL_NUM+int(MODEL_NUM/NUM_ENSEMBLE)+2)]
    sscore = [0 for _ in range(MODEL_NUM+int(MODEL_NUM/NUM_ENSEMBLE)+2)]
    q = Queue()
    while today <= end_date:
        while today.weekday() != 3:
            today = today + datetime.timedelta(days=1)
        today = today + datetime.timedelta(days=1)

        train_bd = today + datetime.timedelta(days=-365*delta_year)
        #train_bd = datetime.date(2011, 1, 1)
        train_ed = today + datetime.timedelta(days=-delta_day)
        test_bd = today + datetime.timedelta(days=1)
        test_ed = today + datetime.timedelta(days=2)
        test_bd_s = "%d%02d%02d" % (test_bd.year, test_bd.month, test_bd.day)
        test_ed_s = "%d%02d%02d" % (test_ed.year, test_ed.month, test_ed.day)
        if not os.path.exists('../txt/1/rcresult/rcresult_1_%s.txt' % test_bd_s) and not os.path.exists('../txt/1/rcresult/rcresult_1_%s.txt' % test_ed_s):
            continue
        p = Process(target=process_train, args=(train_bd, train_ed))
        p.start()
        p.join()
        q.put((sr, sscore))
        p = Process(target=process_test, args=(train_bd, train_ed, q))
        p.start()
        p.join()
        sr, sscore = q.get()




if __name__ == '__main__':
    delta_year = 4
    dbname = '../data/train_201101_20160909.pkl'
    train_bd = datetime.date(2011, 11, 1)
    train_ed = datetime.date(2016, 10, 31)
    test_bd = datetime.date(2016, 6, 5)
    test_ed = datetime.date(2017, 4, 30)

    for delta_year in [6]:
        for nData in [118]:
            simulation_weekly_train0(test_bd, test_ed, 0, delta_year, courses=[0], nData=nData)
            #for c in [1000, 1200, 1300, 1400, 1700]:
            #    for k in [0]:
            #        outfile = '../data/weekly_keras_m1_nd%d_y%d_c%d_0_k%d.txt' % (nData, delta_year, c, k)
            #        simulation_weekly(test_bd, test_ed, outfile, 0, delta_year, c, k, nData=nData)

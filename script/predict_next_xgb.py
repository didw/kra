#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import glob
import parse_xml_entry as xe
import datetime
import train_tf_ensenble as tfn
import train_tf_process as tfp
import train_xgboost as txg
import numpy as np
import os, gzip, cPickle
import itertools
from multiprocessing import Process, Queue
import xgboost as xgb

MODEL_NUM = 10

name_one_hot_columns = ['course', 'humidity', 'kind', 'idx', 'cntry', 'gender', 'age', 'jockey', 'trainer', 'owner', 'cnt', 'rcno', 'month']
def normalize_data(org_data):
    data = org_data.dropna()
    data = data.reset_index()

    column_unique = joblib.load('../data/column_unique.pkl')
    for column in name_one_hot_columns:
        for idx, value in enumerate(sorted(column_unique[column])):
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
            data.loc[i,'jockey'] = -1
        try:
            int(data.loc[i,'trainer'])
        except ValueError:
            data.loc[i,'trainer'] = -1
        try:
            int(data.loc[i,'owner'])
        except ValueError:
            data.loc[i,'owner'] = -1
        i += 1
    return data


"""['course', 'humidity', 'kind', 'rank', 'idx', 'name', 'cntry', 'gender', 'age', 'budam', 'jockey', 'trainer', # 12
  'owner', 'weight', 'dweight', 'rctime', 'r1', 'r2', 'r3', 'cnt', 'rcno', 'month', # 10
  'hr_days', 'hr_nt', 'hr_nt1', 'hr_nt2', 'hr_t1', 'hr_t2', 'hr_ny', 'hr_ny1', 'hr_ny2', 'hr_y1', 'hr_y2', # 11
  'hr_dt', 'hr_d1', 'hr_d2', 'hr_rh', 'hr_rm', 'hr_rl', # 6
  'jk_nt', 'jk_nt1', 'jk_nt2', 'jk_t1', 'jk_t2', 'jk_ny', 'jk_ny1', 'jk_ny2', 'jk_y1', 'jk_y2', # 10
  'tr_nt', 'tr_nt1', 'tr_nt2', 'tr_t1', 'tr_t2', 'tr_ny', 'tr_ny1', 'tr_ny2', 'tr_y1', 'tr_y2'] # 10
  """
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


def get_chulma_fname(date):
    if date.weekday() == 5:
        date_ = date+ datetime.timedelta(days=-2)
    if date.weekday() == 6:
        date_ = date + datetime.timedelta(days=-3)
    return "../txt/1/chulma/chulma_1_%4d%02d%02d.txt" % (date_.year, date_.month, date_.day)


# 300*48 + 2000*6*4 + 14000 + 300*55 = 
def print_detail(players, cand, fresult, mode, total_bet=5000):
    total_num = 0
    for i,j,k in itertools.product(*cand):
        if i!=j and i!=k and j!=k:
            total_num += 1
    bet = max(int(total_bet/total_num/100), 1)*100
    print("bet: %d"%bet)  # 15000 / 5 / 10 = 300
    fresult.write("\n\nbet: %d"%bet)  # 14200 / 6 = 2366
    for i,j,k in itertools.product(*cand):
        if i==j or i==k or j==k:
            continue
        print("%s,%s,%s" % (players[i]+1, players[j]+1, players[k]+1))
        fresult.write("\n%s,%s,%s, %s: %d" % (players[i]+1, players[j]+1, players[k]+1, mode, bet))
    fresult.write("\n")


def print_bet(rcdata, target=[[1],[2],[3]], total_bet=5000):
    global fname
    fresult = open(fname, 'a')
    fresult.write("%s,%s,%s,%s,%s,%s\n" % (rcdata['idx'][0], rcdata['idx'][1], rcdata['idx'][2], rcdata['idx'][3], rcdata['idx'][4], rcdata['idx'][5]))
    print_detail(rcdata['idx'], target, fresult, "ss", total_bet)
    fresult.close()


def predict_next(model_dir, data_pre, meet, date, rcno, course=0, nData=47, year=4, train_course=0, scaler_x1=None, scaler_x2=None, scaler_x3=None, scaler_x4=None, scaler_x5=None, scaler_x6=None, scaler_y=None):
    data = normalize_data(data_pre)
    print(len(data.columns))
    X_data = data.copy()
    print(len(X_data.columns))
    X_data = X_data.drop(['name', 'index'], axis=1)
    __DEBUG__ = True
    if not os.path.exists('../log'):
        os.makedirs('../log')
    if __DEBUG__:
        X_data.to_csv('../log/predict_x_%d_m%d_r%d.csv' % (date, meet, rcno), index=False)
    print(len(X_data.columns))
    X_array = np.array(X_data)
    X_array[:,3:5] = scaler_x1.transform(X_array[:,3:5])
    X_array[:,6:12] = scaler_x2.transform(X_array[:,6:12])
    X_array[:,78:79] = scaler_x3.transform(X_array[:,78:79])
    X_array[:,82:84] = scaler_x4.transform(X_array[:,82:84])
    X_array[:,88:124] = scaler_x5.transform(X_array[:,88:124])
    X_array[:,204:233] = scaler_x6.transform(X_array[:,204:233])

    X_array = xgb.DMatrix(X_array)
    idx_model = 9
    # Loading only for all data trained model
    estimator = joblib.load("%s/%d/model.pkl"% (model_dir, idx_model))
    pred = estimator.predict(X_array)
    pred = pd.DataFrame(pred)
    pred.columns = ['predict']
    __DEBUG__ = True
    if __DEBUG__:
        pd.concat([data_pre, pred], axis=1).to_csv('../log/predict_%d_m%d_r%d.csv' % (date, meet, rcno), index=False)
        X_data.to_csv('../log/predict_x_%d_m%d_r%d.csv' % (date, meet, rcno), index=False)
    prev_rc = data['rcno'][0]
    rcdata = []
    for idx, row in data.iterrows():
        if int(data['hr_nt'][idx]) == 0 or int(data['jk_nt'][idx]) == 0 or int(data['tr_nt'][idx]) == 0:
            print("%s data is not enough. be careful[hr:%d, jk:%d, tr:%d]" % (
                data['name'][idx], int(data['hr_nt'][idx]), int(data['jk_nt'][idx]), int(data['tr_nt'][idx])))
        if row['rcno'] != prev_rc or idx+1 == len(data):
            if idx+1 == len(data):
                rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])
            rcdata = pd.DataFrame(rcdata)
            rcdata.columns = ['idx', 'name', 'time']
            rcdata = rcdata.sort_values(by='time')
            rcdata = rcdata.reset_index(drop=True)
            print("=========== %s ==========" % prev_rc)
            print(rcdata)
            fresult = open(fname, 'a')
            fresult.write("\n\n\n=== rcno: %d, nData: %d, year: %d, train_course: %d ===\n" % (int(prev_rc)+1, nData, year, train_course))
            fresult.close()
            print_bet(rcdata, target=[[1],[2],[3]], total_bet=6000)
            print_bet(rcdata, target=[[1,2,3],[1,2,3],[1,2,3]], total_bet=9000)
            rcdata = []
            prev_rc = row['rcno']
            if idx+1 != len(data):
                rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])
        else:
            rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])


def predict_next_ens(model_dir, data_pre, meet, date, rcno, course=0, nData=47, year=4, train_course=0, 
                    scaler_x1=None, scaler_x2=None, scaler_x3=None, scaler_x4=None, scaler_x5=None, scaler_x6=None, scaler_y=None,
                    idx=1):
    data = normalize_data(data_pre)
    print(len(data.columns))
    X_data = data.copy()
    print(len(X_data.columns))
    X_data = X_data.drop(['name', 'index'], axis=1)
    __DEBUG__ = True
    if not os.path.exists('../log'):
        os.makedirs('../log')
    if __DEBUG__:
        X_data.to_csv('../log/predict_x_%d_m%d_r%d.csv' % (date, meet, rcno), index=False)
    print(len(X_data.columns))
    X_array = np.array(X_data)
    X_array[:,3:5] = scaler_x1.transform(X_array[:,3:5])
    X_array[:,6:12] = scaler_x2.transform(X_array[:,6:12])
    X_array[:,78:79] = scaler_x3.transform(X_array[:,78:79])
    X_array[:,82:84] = scaler_x4.transform(X_array[:,82:84])
    X_array[:,88:124] = scaler_x5.transform(X_array[:,88:124])
    X_array[:,204:233] = scaler_x6.transform(X_array[:,204:233])

    X_array = xgb.DMatrix(X_array)
    idx_model = 0
    for e in range(2):
        preds = [0]*5
        for i in range(5):
            estimator = joblib.load("%s/%d/model.pkl"% (model_dir, idx_model))
            idx_model += 1
            preds[i] = estimator.predict(X_array)

        for i in range(len(preds)+1):
            if i == len(preds):
                pred = np.mean(preds, axis=0)
            else:
                pred = preds[i]
                continue
            pred = pd.DataFrame(pred)
            pred.columns = ['predict']
            __DEBUG__ = True
            if __DEBUG__:
                pd.concat([data_pre, pred], axis=1).to_csv('../log/predict_%d_m%d_r%d_%d.csv' % (date, meet, rcno, i), index=False)
                X_data.to_csv('../log/predict_x_%d_m%d_r%d_%d.csv' % (date, meet, rcno, i), index=False)
            prev_rc = data['rcno'][0]
            rcdata = []
            for idx, row in data.iterrows():
                if int(data['hr_nt'][idx]) == 0 or int(data['jk_nt'][idx]) == 0 or int(data['tr_nt'][idx]) == 0:
                    print("%s data is not enough. be careful[hr:%d, jk:%d, tr:%d]" % (
                        data['name'][idx], int(data['hr_nt'][idx]), int(data['jk_nt'][idx]), int(data['tr_nt'][idx])))
                if row['rcno'] != prev_rc or idx+1 == len(data):
                    if idx+1 == len(data):
                        rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])
                    rcdata = pd.DataFrame(rcdata)
                    rcdata.columns = ['idx', 'name', 'time']
                    rcdata = rcdata.sort_values(by='time')
                    rcdata = rcdata.reset_index(drop=True)
                    print("=========== %s ==========" % prev_rc)
                    print(rcdata)
                    fresult = open(fname, 'a')
                    fresult.write("\n\n\n=== rcno: %d, nData: %d, year: %d, train_course: %d, model: %d ===\n" % (int(prev_rc)+1, nData, year, train_course, i))
                    fresult.close()
                    if idx == 1:
                        print_bet(rcdata, target=[[1,2,3,4],[1,2,3,4],[1,2,3,4]], total_bet=5000)
                    else:
                        print_bet(rcdata, target=[[4,5,6],[4,5,6],[4,5,6,7]], total_bet=5000)
                    rcdata = []
                    prev_rc = row['rcno']
                    if idx+1 != len(data):
                        rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])
                else:
                    rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])


if __name__ == '__main__':
    meet = 1
    train_course = 0
    courses = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    rcno = 0
    #for rcno in range(11, len(courses)):
    course = courses[rcno]
    test_course = course
    init_date = 20171014
    from sklearn.externals import joblib
    md = joblib.load('../data/1_2007_2016_v1_md.pkl')
    with gzip.open('../data/1_2007_2016_v1_md3.gz', 'rb') as f:
        md3 = cPickle.loads(f.read())
    if 25 in md3['humidity']:
        md3['humidity'][20] = md3['humidity'][25]

    data_pre1, data_pre2 = None, None
    data_pre1 = xe.parse_xml_entry(meet, init_date+0, rcno, md, md3)
    data_pre2 = xe.parse_xml_entry(meet, init_date+1, rcno, md, md3)
    for idx in range(1,3):
        nData, year, train_course, epoch = [300,151,201,201][idx-1], [6,6,8,6][idx-1], [0,0,0,0][idx-1], [300,200,200,800][idx-1]
        date = init_date
        if train_course == 1: train_course = course
        print("Process in train: %d, ndata: %d, year: %d" % (train_course, nData, year))

        train_bd = datetime.date(date/10000, date/100%100, date%100) + datetime.timedelta(days=-365*year-1)
        train_ed = datetime.date(date/10000, date/100%100, date%100) + datetime.timedelta(days=-1)
        train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
        train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))
        today_s = "%d_%d"%(train_bd_i, train_ed_i)
        model_base_dir = ["20110226_20170224", today_s][idx-1]
        n_epoch = epoch
        #q = Queue()
        #p = Process(target=txg.training, args=(train_bd, train_ed, q))
        #p.start()
        #p.join()
        #scaler_x1 = q.get()
        #scaler_x2 = q.get()
        #scaler_x3 = q.get()
        #scaler_x4 = q.get()
        #scaler_x5 = q.get()
        #scaler_x6 = q.get()
        #scaler_y = q.get()

        scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_x5, scaler_x6 = joblib.load('../model/xgboost/ens_e1000/%s/scaler_x.pkl' % (model_base_dir,))
        scaler_y = joblib.load('../model/xgboost/ens_e1000/%s/scaler_y.pkl' % (model_base_dir,))
        model_dir = "../model/xgboost/ens_e1000/%s/"%model_base_dir
        if idx in [1,2]:
            fname = '../result/1710/%d_%d.txt' % (date%100, idx)
            os.system("rm %s" % fname)
            predict_next_ens(model_dir, data_pre1, meet, date, rcno, test_course, nData, year, train_course, scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_x5, scaler_x6, scaler_y, idx)
            date += 1
            fname = '../result/1710/%d_%d.txt' % (date%100, idx)
            os.system("rm %s" % fname)
            predict_next_ens(model_dir, data_pre2, meet, date, rcno, test_course, nData, year, train_course, scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_x5, scaler_x6, scaler_y, idx)
        else:
            fname = '../result/1710/%d_%d.txt' % (date%100, idx)
            os.system("rm %s" % fname)
            predict_next(estimators, data_pre1, meet, date, rcno, test_course, nData, year, train_course, scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_x5, scaler_x6, scaler_y)
            date += 1
            fname = '../result/1710/%d_%d.txt' % (date%100, idx)
            os.system("rm %s" % fname)
            predict_next(estimators, data_pre2, meet, date, rcno, test_course, nData, year, train_course, scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_x5, scaler_x6, scaler_y)
        idx += 1

# Strategy
# v1 y6 1,2,3: 10k

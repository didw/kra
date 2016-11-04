#!/usr/bin/python
# -*- coding: utf-8 -*-

import parse_txt_race as pr
import datetime
import pandas as pd
import os.path
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.externals import joblib
import random

def normalize_data(org_data):
    data = org_data.dropna()
    data = data.reset_index()
    data['gender'][data['gender'] == '암'] = 1
    data['gender'][data['gender'] == '수'] = 1
    data['gender'][data['gender'] == '거'] = 2
    data['cntry'][data['cntry'] == '한'] = 0
    data['cntry'][data['cntry'] == '한(포)'] = 1
    data['cntry'][data['cntry'] == '미'] = 2
    data['cntry'][data['cntry'] == '뉴'] = 3
    data['cntry'][data['cntry'] == '호'] = 3
    data['cntry'][data['cntry'] == '일'] = 4
    data['cntry'][data['cntry'] == '캐'] = 5
    data['cntry'][data['cntry'] == '브'] = 6
    data['cntry'][data['cntry'] == '남'] = 6
    data['cntry'][data['cntry'] == '아일'] = 7
    data['cntry'][data['cntry'] == '모'] = 8
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
        filename = "../txt/rcresult/rcresult_1_%02d%02d%02d.txt" % (date.year, date.month, date.day)
        if not os.path.isfile(filename):
            continue
        if first:
            data = pr.get_data(filename)
            first = False
        else:
            data = data.append(pr.get_data(filename), ignore_index=True)
    print(data)
    data = normalize_data(data)
    R_data = data[['rank', 'r1', 'r2', 'r3']]
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
    return X_data, Y_data, R_data, data

# 단승식
def simulation1(pred, ans):
    i = 0
    res = 0
    assert len(pred) == len(ans)
    while True:
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r1 = float(ans['r1'][i])
        i += 1
        total = 1
        while i < len(pred) and int(ans['rank'][i]) != 1:
            sim_data.append(pred[i])
            total += 1
            i += 1
        sim_data = pd.Series(sim_data)
        top = sim_data.argmin()
        #print("prediction: %d" % top)
        if total < 1 or r1 < 1:
            continue
        elif top == 0:
            res += 100 * (r1 - 1)
            print("단승식 WIN: %f" % res)
        else:
            res -= 100
            print("단승식 LOSE: %f" % res)
    return res

# 연승식
def simulation2(pred, ans):
    i = 0
    res = 0
    rcno = 0
    assert len(pred) == len(ans)
    while True:
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r2 = [float(ans['r2'][i])]
        i += 1
        total = 1
        while i < len(pred) and int(ans['rank'][i]) != 1:
            sim_data.append(pred[i])
            r2.append(float(ans['r2'][i]) - 1)
            total += 1
            i += 1
        sim_data = pd.Series(sim_data)
        top = sim_data.argmin()
        #print("prediction: %d" % top)
        rcno += 1
        if total < 1 or r2[top] < 1:
            continue
        elif total > 7:
            if top in [0, 1, 2]:
                res += 100 * r2[top]
                print("연승식(%d) WIN: %f" % (rcno, res))
            else:
                res -= 100
                print("연승식(%d) LOSE: %f" % (rcno, res))
        else:
            if top in [0, 1]:
                res += 100 * r2[top]
                print("연승식(%d) WIN: %f" % (rcno, res))
            else:
                res -= 100
                print("연승식(%d) LOSE: %f" % (rcno, res))
    return res

# 복승식
def simulation3(pred, ans):
    i = 0
    res = 0
    assert len(pred) == len(ans)
    while True:
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r3 = float(ans['r3'][i])
        i += 1
        total = 1
        while i < len(pred) and int(ans['rank'][i]) != 1:
            sim_data.append(pred[i])
            total += 1
            i += 1
        sim_data = pd.Series(sim_data)
        if total < 1 or r3 < 1:
            continue
        top = sim_data.rank()
        if (top[0] in [1, 2]) and (top[1] in [1, 2]):
            print("복승식 WIN: %f = %f + %f" % (res + 100 * r3, res, 100*r3))
            res += 100 * (r3 - 1)
        else:
            res -= 100
            print("복승식 LOSE: %f" % res)
    return res


def simulation_all(pred, ans):
    i = 0
    res = 0
    assert len(pred) == len(ans)
    while True:
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r1 = float(ans['r1'][i]) - 1
        r2 = [float(ans['r2'][i]) - 1]
        r3 = float(ans['r3'][i]) - 1
        i += 1
        total = 1
        while i < len(pred) and int(ans['rank'][i]) != 1:
            sim_data.append(pred[i])
            r2.append(float(ans['r2'][i]) - 1)
            total += 1
            i += 1
        sim_data = pd.Series(sim_data)
        if len(sim_data) < 1:
            continue
        top = sim_data.rank()

        res1 = 100*r1 if top[0] == 1 else -100
        if total > 7:
            res2 = 100*r2[int(top[0]-1)] if top[0] in [1, 2, 3] else -100
        else:
            res2 = 100*r2[int(top[0]-1)] if top[0] in [1, 2] else -100
        res3 = 100*r3 if top[0] in [1, 2] and top[1] in [1, 2] else -100
        res += (res1 + res2 + res3)
        print("res: %f <= (%f) + (%f) + (%f)" % (res, res1, res2, res3))

    return res


def training(bd, ed):
    if os.path.exists('../data/train_data_41.pkl'):
        X_train, Y_train, R_train, _ = joblib.load('../data/train_data_41.pkl')
    else:
        X_train, Y_train, R_train, X_data = get_data(bd, ed)
        joblib.dump([X_train, Y_train, R_train, X_data], '../data/train_data_41.pkl')
    estimator = RandomForestRegressor(random_state=0, n_estimators=100)
    estimator.fit(X_train, Y_train)
    return estimator

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


if __name__ == '__main__':
    #estimator = training(datetime.date(2011, 2, 1), datetime.date(2015, 12, 30), '../model/train_data_41.pkl')
    if os.path.exists('../data/train_data_41.pkl'):
        X_train, Y_train, R_train, _ = joblib.load('../data/train_data_41.pkl')
    else:
        X_train, Y_train, R_train, X_data = get_data(datetime.date(2011, 2, 1), datetime.date(2016, 8, 30))
        joblib.dump([X_train, Y_train, R_train, X_data], '../data/train_data_41.pkl')
    #print X_train
    #print Y_train
    #print R_train

    estimator = RandomForestRegressor(random_state=0, n_estimators=100)
    estimator.fit(X_train, Y_train)
    score = estimator.score(X_train, Y_train)
    print("Score with the entire dataset = %.2f" % score)


    X_test, Y_test, R_test, X_data = get_data(datetime.date(2016, 10, 1), datetime.date(2016, 10, 30))
    score = estimator.score(X_test, Y_test)
    print("Score with the entire dataset = %.2f" % score)
    pred = estimator.predict(X_test)
    __DEBUG__ = False
    if __DEBUG__:
        print_log(X_data, pred, '../log/%s_txt.txt' % "161105")

    res1 = simulation1(pred, R_test)
    res2 = simulation2(pred, R_test)
    res3 = simulation3(pred, R_test)
    res = simulation_all(pred, R_test)

    print("단승식 result: %f" % res1)
    print("연승식 result: %f" % res2)
    print("복승식 result: %f" % res3)
    print("total result: %f" % res)

    import predict_next as pn
    meet = 1
    date = "201610"





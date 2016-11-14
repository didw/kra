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


def get_data_from_csv(begin_date, end_date, fname_csv, course=0):
    df = pd.read_csv(fname_csv)
    remove_index = []
    for idx in range(len(df)):
        #print(df['date'][idx])
        date = int(df['date'][idx])
        if date < begin_date or date > end_date or (course > 0 and course != int(df['course'][idx])):
            remove_index.append(idx)
            #print('Delete %dth row (hr: %s, jk: %s, tr: %s, dt: %s)' % (idx, X_data['hr_nt'][idx], X_data['jk_nt'][idx], X_data['tr_nt'][idx], X_data['hr_dt'][idx]))
    data = df.drop(df.index[remove_index])
    data = normalize_data(data)

    R_data = data[['rank', 'r1', 'r2', 'r3', 'hr_nt', 'hr_dt', 'jk_nt', 'tr_nt', 'cnt', 'rcno', 'price', 'bokyeon1', 'bokyeon2', 'bokyeon3', 'boksik', 'ssang', 'sambok']]
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
    del X_data['price']
    del X_data['bokyeon1']
    del X_data['bokyeon2']
    del X_data['bokyeon3']
    del X_data['boksik']
    del X_data['ssang']
    del X_data['sambok']
    del X_data['weight']
    del X_data['dweight']
    #print(R_data)
    return X_data, Y_data, R_data, data

def delete_lack_data(X_data, Y_data):
    remove_index = []
    for idx in range(len(X_data)):
        if X_data['hr_nt'][idx] == -1 or X_data['jk_nt'][idx] == -1 or X_data['tr_nt'][idx] == -1:
            remove_index.append(idx)
            #print('Delete %dth row (hr: %s, jk: %s, tr: %s, dt: %s)' % (idx, X_data['hr_nt'][idx], X_data['jk_nt'][idx], X_data['tr_nt'][idx], X_data['hr_dt'][idx]))
    print(len(remove_index))
    return X_data.drop(X_data.index[remove_index]), Y_data.drop(Y_data.index[remove_index])


# 단승식
def simulation1(pred, ans):
    #print(ans)
    i = 0
    res1, res2 = 0, 0
    assert len(pred) == len(ans)
    while True:
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r1 = float(ans['r1'][i])
        rcno = int(ans['rcno'][i])
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno:
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            price = int(ans['price'][i])
            total += 1
            i += 1
        a = price*0.8 / r1
        r1 = (price+100000)*0.8 / (a+100000) - 1.0
        if r1 > 20:
            r1 *= 0.8
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top1 = sim_data.argmin()
        top = sim_data.rank()
        if top[0] == 2:
            res2 += 100 * r1
        else:
            res2 -= 100
        if top1 == 1:
            res1 += 100 * r1
        else:
            res1 -= 100
        #print("단승식: %f, %f" % (res1, res2))
    return [res1, res2]

# 연승식
def simulation2(pred, ans):
    print(ans)
    i = 0
    res1 = 0
    res2 = 0
    rcno = 0
    assert len(pred) == len(ans)
    while True:
        rcno += 1
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r2 = [float(ans['r2'][i]) - 1]
        rc_no = int(ans['rcno'][i])
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        while i < len(pred) and int(ans['rcno'][i]) == rc_no:
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            r2.append(float(ans['r2'][i]) - 1)
            total += 1
            i += 1
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top1 = sim_data.argmin()
        top = sim_data.rank()
        if total_player > 7:
            if top1 in [0, 1, 2]:
                if r2[top1] > 20:
                    r2[top1] *= 0.8
                res1 += 100 * r2[top1]
            else:
                res1 -= 100
        else:
            if top1 in [0, 1]:
                if r2[top1] > 20:
                    r2[top1] *= 0.8
                res1 += 100 * r2[top1]
            else:
                res1 -= 100

        if total_player > 7:
            if top[0] == 2:
                res2 += 100 * r2[int(top[0]-1)]
            elif top[1] == 2:
                res2 += 100 * r2[int(top[1]-1)]
            elif top[2] == 2:
                res2 += 100 * r2[int(top[2]-1)]
            else:
                res2 -= 100
        else:
            if top[0] == 2:
                res2 += 100 * r2[int(top[0]-1)]
            elif top[1] == 2:
                res2 += 100 * r2[int(top[1]-1)]
            else:
                res2 -= 100
        print("연승식: %f, %f" % (res1, res2))
    return [res1, res2]

# 복승식
def simulation3(pred, ans):
    print(ans)
    bet = 10
    i = 0
    res1, res2 = 0, 0
    assert len(pred) == len(ans)
    while True:
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r3 = float(ans['r3'][i]) - 1
        if r3*bet > 2000:
            r3 *= 0.8
        rcno = int(ans['rcno'][i])
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno:
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            total += 1
            i += 1
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()
        if total < 2:
            continue
        if (top[0] in [1, 2]) and (top[1] in [1, 2]):
            res1 += bet * r3
        else:
            res1 -= bet

        if total < 2:
            continue
        if (top[0] in [2, 3]) and (top[1] in [2, 3]):
            res2 += bet * r3
        else:
            res2 -= bet
        print("복승식: %f, %f" % (res1, res2))
    return [res1, res2]


def simulation_all(pred, ans):
    print(ans)
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
        rcno = int(ans['rcno'][i])
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno:
            sim_data.append(pred[i])
            r2.append(float(ans['r2'][i]) - 1)
            total_player = int(ans['cnt'][i])
            total += 1
            i += 1
        # if rack_data:
        #     continue
        sim_data = pd.Series(sim_data)
        if total < 2:
            continue
        top = sim_data.rank()
        top1 = sim_data.argmin()

        res1 = 100*r1 if top[0] == 1 else -100
        if total > 7:
            res2 = 100*r2[top1] if top[0] in [1, 2, 3] else -100
        else:
            res2 = 100*r2[top1] if top[0] in [1, 2] else -100
        res3 = 100*r3 if top[0] in [1, 2] and top[1] in [1, 2] else -100
        res += (res1 + res2 + res3)
        print("res: %f <= (%f) + (%f) + (%f)" % (res, res1, res2, res3))

    return res


def training(train_bd, train_ed, course=0):
    train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
    train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))

    model_name = "../model/%d_%d_%s.pkl" % (train_bd_i, train_ed_i, course)

    from sklearn.externals import joblib
    if train_bd < datetime.date.today() + datetime.timedelta(days=-365) and os.path.exists(model_name):
        print("model exist. try to loading..")
        estimator = joblib.load(model_name)
    else:
        print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
        if train_bd >= datetime.date.today() + datetime.timedelta(days=-365):
            X_train, Y_train, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_recent_1year.csv', course)
        else:
            X_train, Y_train, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016.csv', course)
        print("%d data is fully loaded" % len(X_train))

        estimator = RandomForestRegressor(random_state=0, n_estimators=100)
        estimator.fit(X_train, Y_train)

        joblib.dump(estimator, model_name)

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


def simulation_weekly(begin_date, end_date, fname_result, delta_day=0, delta_year=0, course=0):
    today = begin_date
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
        remove_outlier = False
        train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
        train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))

        print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
        X_train, Y_train, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016.csv', course)
        print("%d data is fully loaded" % len(X_train))

        if remove_outlier:
            X_train, Y_train = delete_lack_data(X_train, Y_train)
        print("Start train model")
        estimator = RandomForestRegressor(random_state=0, n_estimators=100)
        estimator.fit(X_train, Y_train)
        print("Finish train model")
        print("important factor")
        #print(X_train.columns)
        #print(estimator.feature_importances_)
        score = estimator.score(X_train, Y_train)
        print("Score with the entire training dataset = %.2f" % score)

        test_bd_i = int("%d%02d%02d" % (test_bd.year, test_bd.month, test_bd.day))
        test_ed_i = int("%d%02d%02d" % (test_ed.year, test_ed.month, test_ed.day))

        print("Loading Datadata at %s - %s" % (str(test_bd), str(test_ed)))
        X_test, Y_test, R_test, X_data = get_data_from_csv(test_bd_i, test_ed_i, '../data/1_2007_2016.csv', course)
        print("%d data is fully loaded" % (len(X_test)))
        if len(X_test) == 0:
            res1 = [0, 0]
        else:
            DEBUG = False
            if DEBUG:
                X_test.to_csv('../log/2016_7_9.csv', index=False)
            score = estimator.score(X_test, Y_test)
            print("Score with the entire test dataset = %.2f" % score)
            pred = estimator.predict(X_test)

            res1 = simulation1(pred, R_test)

        print("train data: %s - %s\n" % (str(train_bd), str(train_ed)))
        print("test data: %s - %s\n" % (str(test_bd), str(test_ed)))
        print("course: %d(0: all)\n" % course)
        print("단승식 result: %f\n\n" % (res1[0]))
        f_result = open(fname_result, 'a')
        f_result.write("train data: %s - %s\n" % (str(train_bd), str(train_ed)))
        f_result.write("test data: %s - %s\n" % (str(test_bd), str(test_ed)))
        f_result.write("단승식 result: %f\n\n" % (res1[0]))
        f_result.close()


if __name__ == '__main__':
    delta_year = 2
    dbname = '../data/train_201101_20160909.pkl'
    train_bd = datetime.date(2011, 11, 1)
    train_ed = datetime.date(2016, 10, 31)
    test_bd = datetime.date(2016, 1, 1)
    test_ed = datetime.date(2016, 11, 12)
    for c in [1000, 1200, 1300, 1400, 1700, 2000, 0]:
        outfile = '../data/weekly_result_m1_y%d_c%d.txt' % (delta_year, c)
        simulation_weekly(test_bd, test_ed, outfile, 0, delta_year, c)
    remove_outlier = False
"""
    #estimator = training(datetime.date(2011, 2, 1), datetime.date(2015, 12, 30))
    if os.path.exists(dbname):
        X_train, Y_train = joblib.load(dbname)
    else:
        X_train, Y_train, _, _ = get_data(train_bd, train_ed)
        joblib.dump([X_train, Y_train], dbname)

    if remove_outlier:
        print(len(X_train))
        X_train, Y_train = delete_lack_data(X_train, Y_train)
        print(len(X_train))
    estimator = RandomForestRegressor(random_state=0, n_estimators=100)
    estimator.fit(X_train, Y_train)
    print("important factor")
    print(X_train.columns)
    print(estimator.feature_importances_)
    score = estimator.score(X_train, Y_train)
    print("Score with the entire training dataset = %.2f" % score)

    X_test, Y_test, R_test, X_data = get_data(test_bd, test_ed)
    DEBUG = False
    if DEBUG:
        X_test.to_csv('../log/2016_7_9.csv', index=False)
    score = estimator.score(X_test, Y_test)
    print("Score with the entire test dataset = %.2f" % score)
    pred = estimator.predict(X_test)

    res1 = simulation1(pred, R_test)
    res2 = simulation2(pred, R_test)
    res3 = simulation3(pred, R_test)
    res = simulation_all(pred, R_test)

    print("db name: %s" % dbname)
    print("remove_outlier: %s" % remove_outlier)
    print("train data: %s - %s" % (str(train_bd), str(train_ed)))
    print("test data: %s - %s" % (str(test_bd), str(test_ed)))
    print("단승식 result: %f, %f" % (res1[0], res1[1]))
    print("연승식 result: %f, %f" % (res2[0], res2[1]))
    print("복승식 result: %f, %f" % (res3[0], res3[1]))
    print("total result: %f" % res)

"""
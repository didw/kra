#!/usr/bin/python
# -*- coding: utf-8 -*-

import parse_txt_race as pr
import datetime
import pandas as pd
import os.path
from sklearn.ensemble.forest import RandomForestRegressor

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
    while date < train_ed:
        date = date + datetime.timedelta(days=1)
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
    del data['name']
    del data['jockey']
    del data['trainer']
    del data['owner']
    data = normalize_data(data)
    R_data = data[['rank', 'r1', 'r2', 'r3']]
    Y_data = data['rctime']
    X_data = data
    del X_data['rctime']
    del X_data['rank']
    del X_data['r3']
    del X_data['r2']
    del X_data['r1']
    return X_data, Y_data, R_data

# 단승식
def simulation1(pred, ans):
    i = 0
    res = 0
    assert len(pred) == len(ans)
    print(pred)
    print(ans)
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
        if top == 0:
            res += 100 * r1
            print("단승식 WIN: %f" % res)
        else:
            res -= 100
            print("단승식 LOSE: %f" % res)
    return res

# 연승식
def simulation2(pred, ans):
    i = 0
    res = 0
    assert len(pred) == len(ans)
    print(pred)
    print(ans)
    while True:
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r2 = float(ans['r2'][i])
        i += 1
        total = 1
        while i < len(pred) and int(ans['rank'][i]) != 1:
            sim_data.append(pred[i])
            total += 1
            i += 1
        sim_data = pd.Series(sim_data)
        top = sim_data.argmin()
        #print("prediction: %d" % top)
        if total > 7:
            if top == 0 or top == 1 or top == 2:
                res += 100 * r2
                print("WIN: %f" % res)
            else:
                res -= 100
                print("LOSE: %f" % res)
        else:
            if top == 0 or top == 1:
                res += 100 * r2
                print("연승식 WIN: %f" % res)
            else:
                res -= 100
                print("연승식 LOSE: %f" % res)
    return res

# 복승식
def simulation3(pred, ans):
    i = 0
    res = 0
    assert len(pred) == len(ans)
    print(pred)
    print(ans)
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
        top = sim_data.rank()[0]
        #print("prediction: %d" % top)
        if (top[0] == 0 or top[0] == 1) and (top[1] == 0 or top[1] == 1):
            res += 100 * r3
            print("복승식 WIN: %f" % res)
        else:
            res -= 100
            print("복승식 LOSE: %f" % res)
    return res

X_train, Y_train, R_train = get_data(datetime.date(2011, 2, 1), datetime.date(2015, 12, 30))
#print X_train
#print Y_train
#print R_train

estimator = RandomForestRegressor(random_state=0, n_estimators=100)
estimator.fit(X_train, Y_train)
score = estimator.score(X_train, Y_train)
print("Score with the entire dataset = %.2f" % score)


X_test, Y_test, R_test = get_data(datetime.date(2016, 1, 1), datetime.date(2016, 9, 30))
score = estimator.score(X_test, Y_test)
print("Score with the entire dataset = %.2f" % score)
pred = estimator.predict(X_test)
res1 = simulation1(pred, R_test)
res2 = simulation2(pred, R_test)
res3 = simulation3(pred, R_test)

print("단승식 result: %f" % res1)
print("연승식 result: %f" % res2)
print("복승식 result: %f" % res3)




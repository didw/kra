#!/usr/bin/python
# -*- coding: utf-8 -*-

import parse_txt_race as pr
import datetime
import pandas as pd
import os.path
from sklearn.ensemble.forest import RandomForestRegressor

def normalize_data(data):
    data['gender'][data['gender'] == '암'] = 1
    data['gender'][data['gender'] == '수'] = 1
    data['gender'][data['gender'] == '거'] = 2
    data['cntry'][data['cntry'] == '한'] = 0
    data['cntry'][data['cntry'] == '미'] = 1
    data['cntry'][data['cntry'] == '뉴'] = 2
    data['cntry'][data['cntry'] == '호'] = 3
    data['cntry'][data['cntry'] == '한(포)'] = 4
    data['cntry'][data['cntry'] == '일'] = 5
    data['cntry'][data['cntry'] == '캐'] = 6

train_bd = datetime.date(2016, 1, 1)
train_ed = datetime.date(2016, 5, 10)

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

del data['rank']
del data['name']
del data['jockey']
del data['trainer']
del data['owner']

normalize_data(data)

Y_train = data['rctime']
X_train = data
del X_train['rctime']

print(X_train)
print(Y_train)

estimator = RandomForestRegressor(random_state=0, n_estimators=100)
estimator.fit(X_train, Y_train)
score = estimator.score(X_train, Y_train)
print("Score with the entire dataset = %.2f" % score)



pred = estimator.predict(X_train)
print(pred)

error = pred - Y_train

print(error)



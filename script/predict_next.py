#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob
import parse_xml_entry as xe
import datetime
import train as tr

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


def predict_next(estimator, meet, date):
    data = xe.parse_xml_entry(meet, date)
    data = normalize_data(data)
    X_data = data.copy()
    del X_data['name']
    del X_data['jockey']
    del X_data['trainer']
    del X_data['owner']
    del X_data['rcno']
    pred = pd.DataFrame(estimator.predict(X_data))
    pred.columns = ['predict']
    __DEBUG__ = True
    if __DEBUG__:
        fdata = open('../log/161031.txt', 'w')
        for idx, row in data.iterrows():
            for item in data.columns:
                fdata.write("%s\t" % row[item])
            fdata.write("%f\n" % pred['predict'][idx])
        fdata.close()
    print(pd.concat([data, pred], axis=1))
    print(pd.concat([data[['rcno', 'name', 'jockey', 'trainer']], pred], axis=1))
    prev_rc = data['rcno'][0]
    rctime = []
    rcdata = []
    for idx, row in data.iterrows():
        if row['rcno'] != prev_rc or idx+1 == len(data):
            rctime = pd.Series(rctime)
            rcdata = pd.DataFrame(rcdata)
            rcrank = rctime.rank()
            for i, v in enumerate(rcrank):
                if v == 1:
                    print("rcNo: %s, 1st: %s (%s): %f" % (rcdata[0][i], rcdata[2][i], rcdata[1][i], rctime[i]))
                elif v == 2:
                    print("rcNo: %s, 2nd: %s (%s): %f" % (rcdata[0][i], rcdata[2][i], rcdata[1][i], rctime[i]))
            rctime = []
            rcdata = []
            prev_rc = row['rcno']
        else:
            rctime.append(float(pred['predict'][idx]))
            rcdata.append([row['rcno'], row['name'], row['idx']])


if __name__ == '__main__':
    meet = 1
    date = 20161030
    import get_api
    #get_api.get_data(meet, date/100)
    estimator = tr.training(datetime.date(2011, 2, 1), datetime.date(2016, 10, 25))
    predict_next(estimator, meet, date)



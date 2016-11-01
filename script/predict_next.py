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
    pred = estimator.predict(X_data)
    prev_rc = data['rcno'][0]
    for idx, row in data.iterrows():
        rctime = []
        rcdata = []
        if row['rcno'] == prev_rc:
            rctime.append(pred[idx])
            rcdata.append([row['rcno'], row['name'], data['idx']])
        else:
            rctime = pd.Series(rctime)
            rcdata = pd.DataFrame(rcdata)
            rcrank = rctime.rank()
            for i, v in enumerate(rcrank):
                if v == 1:
                    print("rcNo: %s, 1st: %d (%s)" % (rcdata['rcno'][i], rcdata['idx'][i], rcdata['name'][i]))
                elif v == 2:
                    print("rcNo: %s, 2nd: %d (%s)" % (rcdata['rcno'][i], rcdata['idx'][i], rcdata['name'][i]))


if __name__ == '__main__':
    meet = 1
    date = 20161030
    import get_api
    #get_api.get_data(meet, date/100)
    estimator = tr.training(datetime.date(2011, 2, 1), datetime.date(2016, 10, 31))
    estimator = None
    predict_next(estimator, meet, date)



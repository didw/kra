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


"""['course', 'humidity', 'idx', 'name', 'cntry', 'gender', 'age', 'budam', 'jockey', 'trainer', 'owner',
 'weight', 'dweight', 'hr_days', 'hr_nt', 'hr_nt1', 'hr_nt2', 'hr_t1', 'hr_t2', 'hr_ny', 'hr_ny1',
 'hr_ny2', 'hr_y1', 'hr_y2', 'jk_nt', 'jk_nt1', 'jk_nt2', 'jk_t1', 'jk_t2', 'jk_ny', 'jk_ny1',
 'jk_ny2', 'jk_y1', 'jk_y2', 'tr_nt', 'tr_nt1', 'tr_nt2', 'tr_t1', 'tr_t2', 'tr_ny', 'tr_ny1',
 'tr_ny2', 'tr_y1', 'tr_y2', 'rcno']
 """
def print_log(data, pred, fname):
    flog = open(fname, 'w')
    rcno = 1
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
        print_log(data, pred, '../log/%s.txt' % data)
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
    date = 20161105
    import get_api
    get_api.get_data(meet, date/100)
    estimator = tr.training(datetime.date(2011, 2, 1), datetime.date(2016, 10, 25))
    predict_next(estimator, meet, date)



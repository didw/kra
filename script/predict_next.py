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


def predict_next(estimator, meet, date, rcno):
    data_pre = xe.parse_xml_entry(meet, date, rcno)
    data = normalize_data(data_pre)
    X_data = data.copy()
    del X_data['name']
    del X_data['jockey']
    del X_data['trainer']
    del X_data['owner']
    pred = pd.DataFrame(estimator.predict(X_data))
    pred.columns = ['predict']
    __DEBUG__ = True
    if __DEBUG__:
        pd.concat([data_pre, pred], axis=1).to_csv('../log/%d_xml.csv' % date, index=False)
    #print(pd.concat([data, pred], axis=1))
    #print(pd.concat([data[['rcno', 'name', 'jockey', 'trainer']], pred], axis=1))
    prev_rc = data['rcno'][0]
    rctime = []
    rcdata = []
    for idx, row in data.iterrows():
        if int(data['hr_nt'][idx]) == 0 or int(data['jk_nt'][idx]) == 0 or int(data['tr_nt'][idx]) == 0:
            print("%s data is not enough. be careful" % (data['name'][idx]))
        if row['rcno'] != prev_rc or idx+1 == len(data):
            rctime = pd.Series(rctime)
            rcdata = pd.DataFrame(rcdata)
            rcrank = rctime.rank()
            print("rcNo, order, idx(name): rctime")
            for i, v in enumerate(rcrank):
                if v == 1:
                    print("%s, 1st: %s (%s): %f" % (rcdata[0][i], rcdata[2][i], rcdata[1][i], rctime[i]))
                elif v == 2:
                    print("%s, 2nd: %s (%s): %f" % (rcdata[0][i], rcdata[2][i], rcdata[1][i], rctime[i]))
                elif v == 3:
                    print("%s, 3rd: %s (%s): %f" % (rcdata[0][i], rcdata[2][i], rcdata[1][i], rctime[i]))
            print("")
            rctime = []
            rcdata = []
            prev_rc = row['rcno']
        else:
            rctime.append(float(pred['predict'][idx]))
            rcdata.append([row['rcno'], row['name'], row['idx']])


if __name__ == '__main__':
    meet = 1
    date = 20161113
    rcno = 11
    #import get_api
    #get_api.get_data(meet, date/100)
    estimator1 = tr.training(datetime.date(2011, 1, 1), datetime.date(2016, 10, 31))
    estimator2 = tr.training(datetime.date(2011, 11, 1), datetime.date(2016, 10, 31))
    estimator3 = tr.training(datetime.date(2015, 11, 1), datetime.date(2016, 10, 31))
    import get_txt
    get_txt.download_txt(datetime.date.today(), datetime.date.today(), 1)
    estimator4 = tr.training(datetime.date.today() + datetime.timedelta(days=-365), datetime.date.today())

    predict_next(estimator1, meet, date, rcno)
    predict_next(estimator2, meet, date, rcno)
    predict_next(estimator3, meet, date, rcno)
    predict_next(estimator4, meet, date, rcno)



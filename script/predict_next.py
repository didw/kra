#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob
import parse_xml_entry as xe
import datetime
import train as tr


def normalize_data(org_data, nData=47):
    data = org_data.dropna()
    data = data.reset_index()
    data.loc[data['gender'] == '암', 'gender'] = 0
    data.loc[data['gender'] == '수', 'gender'] = 1
    data.loc[data['gender'] == '거', 'gender'] = 2
    data.loc[data['cntry'] == '한', 'cntry'] = 0
    data.loc[data['cntry'] == '한(포)', 'cntry'] = 1
    data.loc[data['cntry'] == '제', 'cntry'] = 2
    data.loc[data['cntry'] == '일', 'cntry'] = 3
    data.loc[data['cntry'] == '중', 'cntry'] = 4
    data.loc[data['cntry'] == '미', 'cntry'] = 5
    data.loc[data['cntry'] == '캐', 'cntry'] = 6
    data.loc[data['cntry'] == '뉴', 'cntry'] = 7
    data.loc[data['cntry'] == '호', 'cntry'] = 8
    data.loc[data['cntry'] == '브', 'cntry'] = 9
    data.loc[data['cntry'] == '헨', 'cntry'] = 10
    data.loc[data['cntry'] == '남', 'cntry'] = 11
    data.loc[data['cntry'] == '아일', 'cntry'] = 12
    data.loc[data['cntry'] == '모', 'cntry'] = 13
    data.loc[data['cntry'] == '영', 'cntry'] = 14
    data.loc[data['cntry'] == '인', 'cntry'] = 15
    data.loc[data['cntry'] == '아', 'cntry'] = 16
    data.loc[data['cntry'] == '프', 'cntry'] = 17
    if nData == 47:
        del data['ts1']
        del data['ts2']
        del data['ts3']
        del data['ts4']
        del data['ts5']
        del data['ts6']
        del data['score1']
        del data['score2']
        del data['score3']
        del data['score4']
        del data['score5']
        del data['score6']
        del data['score7']
        del data['score8']
        del data['score9']
        del data['score10']
        del data['hr_dt']
        del data['hr_d1']
        del data['hr_d2']
        del data['hr_rh']
        del data['hr_rm']
        del data['hr_rl']

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
    if date.weekday() == 4:
        date_ = date+ datetime.timedelta(days=-2)
    if date.weekday() == 5:
        date_ = date + datetime.timedelta(days=-3)
    return "../txt/2/chulma/chulma_2_%4d%02d%02d.txt" % (date_.year, date_.month, date_.day)


# 300*48 + 2000*6*4 + 14000 + 300*55 = 
def print_detail(players, cand, fresult):
    if cand == [[1,2,3],[1,2,3,4,5],[1,2,3,4,5,6]]:
        print("bet: 100") # 14200 / 48 = 295
        print("%s,%s,{%s,%s,%s,%s}" % (players[0], players[1], players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[0], players[2], players[1], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[0], players[3], players[1], players[2], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[0], players[4], players[1], players[2], players[3], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[1], players[0], players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[1], players[2], players[0], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[1], players[3], players[0], players[2], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[1], players[4], players[0], players[2], players[3], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[2], players[0], players[1], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[2], players[1], players[0], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[2], players[3], players[0], players[1], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[2], players[4], players[0], players[1], players[3], players[5]))
        fresult.write("\n\nbet: 100") # 14200 / 48 = 295
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[0], players[1], players[2], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[0], players[2], players[1], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[0], players[3], players[1], players[2], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[0], players[4], players[1], players[2], players[3], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[1], players[0], players[2], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[1], players[2], players[0], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[1], players[3], players[0], players[2], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[1], players[4], players[0], players[2], players[3], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[2], players[0], players[1], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[2], players[1], players[0], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[2], players[3], players[0], players[1], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[2], players[4], players[0], players[1], players[3], players[5]))
    elif cand == [[4,5,6],[4,5,6],[4,5,6]]:
        print("bet: 700")  # 14200 / 6 = 2366
        print("%s,%s,%s" % (players[3], players[4], players[5]))
        print("%s,%s,%s" % (players[3], players[5], players[4]))
        print("%s,%s,%s" % (players[4], players[3], players[5]))
        print("%s,%s,%s" % (players[4], players[5], players[3]))
        print("%s,%s,%s" % (players[5], players[3], players[4]))
        print("%s,%s,%s" % (players[5], players[4], players[3]))

        fresult.write("\n\nbet: 700")  # 14200 / 6 = 2366
        fresult.write("\n%s,%s,%s" % (players[3], players[4], players[5]))
        fresult.write("\n%s,%s,%s" % (players[3], players[5], players[4]))
        fresult.write("\n%s,%s,%s" % (players[4], players[3], players[5]))
        fresult.write("\n%s,%s,%s" % (players[4], players[5], players[3]))
        fresult.write("\n%s,%s,%s" % (players[5], players[3], players[4]))
        fresult.write("\n%s,%s,%s" % (players[5], players[4], players[3]))
    elif cand == [[4,5,6,7,8],[4,5,6,7,8],[4,5,6,7,8]]:
        print("bet: 100")  # 14200 / 60 = 200
        print("%s,%s,{%s,%s,%s}" % (players[3], players[4], players[5], players[6], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[3], players[5], players[4], players[6], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[3], players[6], players[4], players[5], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[3], players[7], players[4], players[5], players[6]))
        print("%s,%s,{%s,%s,%s}" % (players[4], players[3], players[5], players[6], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[4], players[5], players[3], players[6], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[4], players[6], players[3], players[5], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[4], players[7], players[3], players[5], players[6]))
        print("%s,%s,{%s,%s,%s}" % (players[5], players[3], players[4], players[6], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[5], players[4], players[3], players[6], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[5], players[6], players[3], players[4], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[5], players[7], players[3], players[4], players[6]))
        print("%s,%s,{%s,%s,%s}" % (players[6], players[3], players[4], players[5], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[6], players[4], players[3], players[5], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[6], players[5], players[3], players[4], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[6], players[7], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[7], players[3], players[4], players[5], players[6]))
        print("%s,%s,{%s,%s,%s}" % (players[7], players[4], players[3], players[5], players[6]))
        print("%s,%s,{%s,%s,%s}" % (players[7], players[5], players[3], players[4], players[6]))
        print("%s,%s,{%s,%s,%s}" % (players[7], players[6], players[3], players[4], players[5]))

        fresult.write("\n\nbet: 100")  # 14200 / 60 = 200
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[3], players[4], players[5], players[6], players[7]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[3], players[5], players[4], players[6], players[7]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[3], players[6], players[4], players[5], players[7]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[3], players[7], players[4], players[5], players[6]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[4], players[3], players[5], players[6], players[7]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[4], players[5], players[3], players[6], players[7]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[4], players[6], players[3], players[5], players[7]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[4], players[7], players[3], players[5], players[6]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[5], players[3], players[4], players[6], players[7]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[5], players[4], players[3], players[6], players[7]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[5], players[6], players[3], players[4], players[7]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[5], players[7], players[3], players[4], players[6]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[6], players[3], players[4], players[5], players[7]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[6], players[4], players[3], players[5], players[7]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[6], players[5], players[3], players[4], players[7]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[6], players[7], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[7], players[3], players[4], players[5], players[6]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[7], players[4], players[3], players[5], players[6]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[7], players[5], players[3], players[4], players[6]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[7], players[6], players[3], players[4], players[5]))
    elif cand == [[1],[2],[3]]:
        print("bet: 4000")  # 14200
        print("%s,%s,%s" % (players[0], players[1], players[2]))
        fresult.write("\n\nbet: 4000")  # 14200
        fresult.write("\n%s,%s,%s" % (players[0], players[1], players[2]))
    elif cand == [[1,2,3,4],[1,2,3,4,5,6],[3,4,5,6]]:
        print("bet: 100") # 14200 / 55 = 258
        print("%s,%s,{%s,%s,%s,%s}" % (players[0], players[1], players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[0], players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[0], players[3], players[2], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[0], players[4], players[2], players[3], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[0], players[5], players[2], players[3], players[4]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[1], players[0], players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[1], players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[1], players[3], players[2], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[1], players[4], players[2], players[3], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[1], players[5], players[2], players[3], players[4]))
        print("%s,%s,{%s,%s,%s}" % (players[2], players[0], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[2], players[1], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s}" % (players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s}" % (players[2], players[4], players[3], players[5]))
        print("%s,%s,{%s,%s}" % (players[2], players[5], players[3], players[4]))
        print("%s,%s,{%s,%s,%s}" % (players[3], players[0], players[2], players[4], players[5]))
        print("%s,%s,{%s,%s}" % (players[3], players[1], players[4], players[5]))
        print("%s,%s,{%s,%s}" % (players[3], players[2], players[4], players[5]))
        print("%s,%s,{%s,%s}" % (players[3], players[4], players[2], players[5]))
        print("%s,%s,{%s,%s}" % (players[3], players[5], players[2], players[4]))

        fresult.write("\n\nbet: 100")  # 14200 / 55 = 258
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[0], players[1], players[2], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[0], players[2], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[0], players[3], players[2], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[0], players[4], players[2], players[3], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[0], players[5], players[2], players[3], players[4]))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}" % (players[1], players[0], players[2], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[1], players[2], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[1], players[3], players[2], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[1], players[4], players[2], players[3], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[1], players[5], players[2], players[3], players[4]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[2], players[0], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[2], players[1], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s}" % (players[2], players[3], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s}" % (players[2], players[4], players[3], players[5]))
        fresult.write("\n%s,%s,{%s,%s}" % (players[2], players[5], players[3], players[4]))
        fresult.write("\n%s,%s,{%s,%s,%s}" % (players[3], players[0], players[2], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s}" % (players[3], players[1], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s}" % (players[3], players[2], players[4], players[5]))
        fresult.write("\n%s,%s,{%s,%s}" % (players[3], players[4], players[2], players[5]))
        fresult.write("\n%s,%s,{%s,%s}" % (players[3], players[5], players[2], players[4]))


def print_bet(rcdata, course=0, year=4, nData=47, train_course=0):
    print("dan")
    print("%s" % (rcdata['idx'][0]))
    print("bok")
    print("%s,%s" % (rcdata['idx'][0], rcdata['idx'][1]))
    print("bokyeon")
    print("%s,%s" % (rcdata['idx'][0], rcdata['idx'][1]))
    print("ssang")
    print("%s,{%s,%s}" % (rcdata['idx'][0], rcdata['idx'][1], rcdata['idx'][2]))
    print("%s,{%s,%s}" % (rcdata['idx'][1], rcdata['idx'][0], rcdata['idx'][2]))

    print("sambok")
    print("%s,%s,{%s,%s}" % (rcdata['idx'][0], rcdata['idx'][1], rcdata['idx'][2], rcdata['idx'][3]))
    print("{%s,%s},%s,%s" % (rcdata['idx'][0], rcdata['idx'][1], rcdata['idx'][2], rcdata['idx'][3]))

    print("samssang")
    global fname
    fresult = open(fname, 'a')
    print_detail(rcdata['idx'], [[1],[2],[3]], fresult)

    fresult.close()


def predict_next(estimator, md, rd, meet, date, rcno, course=0, nData=47, year=4, train_course=0):
    data_pre = xe.parse_xml_entry(meet, date, rcno, md, rd)
    data = normalize_data(data_pre, nData=nData)
    print(len(data.columns))
    X_data = data.copy()
    print(len(X_data.columns))
    del X_data['name']
    del X_data['jockey']
    del X_data['trainer']
    del X_data['owner']
    del X_data['index']
    __DEBUG__ = True
    if __DEBUG__:
        X_data.to_csv('../log/predict_x_%d_m%d_r%d.csv' % (date, meet, rcno), index=False)
    pred = pd.DataFrame(estimator.predict(X_data))
    pred.columns = ['predict']
    __DEBUG__ = True
    if __DEBUG__:
        pd.concat([data_pre, pred], axis=1).to_csv('../log/predict_%d_m%d_r%d.csv' % (date, meet, rcno), index=False)
        X_data.to_csv('../log/predict_x_%d_m%d_r%d.csv' % (date, meet, rcno), index=False)
    prev_rc = data['rcno'][0]
    rcdata = []
    for idx, row in data.iterrows():
        if int(data['hr_nt'][idx]) == 0 or int(data['jk_nt'][idx]) == 0 or int(data['tr_nt'][idx]) == 0:
            print("%s data is not enough. be careful" % (data['name'][idx]))
        if row['rcno'] != prev_rc or idx+1 == len(data):
            if idx+1 == len(data):
                rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])
            rcdata = pd.DataFrame(rcdata)
            rcdata.columns = ['idx', 'name', 'time']
            rcdata = rcdata.sort_values(by='time')
            rcdata = rcdata.reset_index(drop=True)
            print("=========== %s ==========" % prev_rc)
            print(rcdata)
            print_bet(rcdata, course, nData=nData, year=year, train_course=train_course)
            rcdata = []
            prev_rc = row['rcno']
            if idx+1 != len(data):
                rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])
        else:
            rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])
    print(X_data.columns)
    print(estimator.feature_importances_)

def get_race_detail(date):
    rd = RaceDetail()
    import glob
    for year in range(date/10000 - 3, date/10000):
        filelist1 = glob.glob('../txt/2/ap-check-rslt/ap-check-rslt_2_%d*.txt' % year)
        filelist2 = glob.glob('../txt/2/rcresult/rcresult_2_%d*.txt' % year)
        print("loading rslt at %d" % year)
        for fname in filelist1:
            rd.parse_ap_rslt(fname)
        print("loading rcresult at %d" % year)
        for fname in filelist2:
            rd.parse_race_detail(fname)
    return rd

if __name__ == '__main__':
    meet = 2
    date = 20161218
    rcno = 11
    train_course = 0
    courses = [1000, 1300, 1300, 1200, 1300, 1300, 1300, 1700, 1700, 1800, 1200]
    rcno = 1
    #for rcno in range(len(courses)):
    course = courses[rcno-1]
    test_course = course
    rd = get_race_detail(date)
    fname = '../result/1701/%d_%d.txt' % (date%100, rcno)
    for nData, year in zip([186], [2]):
        print("Process in train: %d, ndata: %d, year: %d" % (train_course, nData, year))
        estimator, md, umd = tr.training(datetime.date(date/10000, date/100%100, date%100) + datetime.timedelta(days=-365*year), datetime.date(date/10000, date/100%100, date%100), train_course, nData)
        predict_next(estimator, md, rd, meet, date, rcno, test_course, nData, year, train_course)
#        train_course = course
#        for nData in [186]:
#            for year in [2]:
#                print("Process in train: %d, ndata: %d, year: %d" % (train_course, nData, year))
#                estimator, md, umd = tr.training(datetime.date(date/10000, date/100%100, date%100) + datetime.timedelta(days=-365*year), datetime.date(date/10000, date/100%100, date%100), train_course, nData)
#                predict_next(estimator, md, rd, meet, date, rcno, test_course, nData, year, train_course)

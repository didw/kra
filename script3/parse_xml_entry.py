# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob
import re
import sys
import parse_xml_jk as xj
import parse_xml_hr as xh
import parse_xml_tr as xt
import parse_xml_train as xtr
import datetime
import os
import get_detail_data as gdd
import get_lineage as gl
from mean_data import mean_data
import get_weekly_clinic as wc
from race_record import RaceRecord
import gzip, _pickle
import requests



DEBUG = False

def get_humidity():
    url = "http://race.kra.co.kr/chulmainfo/trackView.do?Act=02&Sub=10&meet=1"
    response_body = requests.get(url)
    line = unicode(response_body.text, 'euc-kr').encode('utf-8')
    #print("%s" % line)
    p = re.compile(unicode(r'(?<=함수율 <span>: )\d+(?=\%\()', 'utf-8').encode('utf-8'), re.MULTILINE)
    pl = p.search(line)
    res = 10
    if pl is not None:
        res = pl.group()
    return res


def get_hr_data(data, name):
    name = name.replace('★'.encode('utf-8'), '')
    for idx, line in data.iterrows():
        #print ("line: ", line)
        #print ("name: %s" % name)
        if line['hrName'] == name:
            hr_birth = line['birth']
            hr_birth = (datetime.date.today() - datetime.date(int(hr_birth[:4]), int(hr_birth[5:7]), int(hr_birth[8:]))).days
            return (line['gender'], hr_birth)
    print("can not find horse %s" % name.encode('utf-8'))
    return (-1, -1)


def get_hr_win(tt, t1, t2, yt, y1, y2, course, md=mean_data()):
    res = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    if int(tt) != 0:
        res[0], res[1], res[2] = int(tt), int(t1), int(t2)
        res[3] = float(t1) * 100 / int(tt)
        res[4] = float(t2) * 100 / int(tt)
    else:
        res[1] = md.hr_history_total[course][1]
        res[2] = md.hr_history_total[course][2]
        res[3] = md.hr_history_total[course][3]
        res[4] = md.hr_history_total[course][4]
    if int(yt) != 0:
        res[5], res[6], res[7] = int(yt), int(y1), int(y2)
        res[8] = float(y1) * 100 / int(yt)
        res[9] = float(y2) * 100 / int(yt)
    else:
        res[6] = md.hr_history_year[course][1]
        res[7] = md.hr_history_year[course][2]
        res[8] = md.hr_history_year[course][3]
        res[9] = md.hr_history_year[course][4]
    return res


def get_jk_win(data, name, course, md=mean_data()):
    res = md.jk_history_total[course] + md.jk_history_year[course]
    for idx, line in data.iterrows():
        if line['jkName'] == name:
            res[0] = tt = float(line['cntT'])
            res[1] = t1 = float(line['ord1T'])
            res[2] = t2 = float(line['ord2T'])
            res[5] = yt = float(line['cntY'])
            res[6] = y1 = float(line['ord1Y'])
            res[7] = y2 = float(line['ord2Y'])
            if int(tt) != 0:
                res[3] = float(t1) * 100 / float(tt)
                res[4] = float(t2) * 100 / float(tt)
            else:
                res[3] = md.jk_history_total[course][3]
                res[4] = md.jk_history_total[course][4]
            if int(yt) != 0:
                res[8] = float(y1) * 100 / float(yt)
                res[9] = float(y2) * 100 / float(yt)
            else:
                res[8] = md.jk_history_year[course][3]
                res[9] = md.jk_history_year[course][4]
            return res
    print("can not find jockey %s" % (name,))
    return res


def get_tr_win(data, name, course, md=mean_data()):
    res = md.tr_history_total[course] + md.tr_history_year[course]
    for idx, line in data.iterrows():
        if line['trName'] == name:
            res[0] = tt = float(line['cntT'])
            res[1] = t1 = float(line['ord1T'])
            res[2] = t2 = float(line['ord2T'])
            res[5] = yt = float(line['cntY'])
            res[6] = y1 = float(line['ord1Y'])
            res[7] = y2 = float(line['ord2Y'])
            if int(tt) != 0:
                res[3] = float(t1) * 100 / float(tt)
                res[4] = float(t2) * 100 / float(tt)
            else:
                res[3] = md.tr_history_total[course][3]
                res[4] = md.tr_history_total[course][4]
            if int(yt) != 0:
                res[8] = float(y1) * 100 / float(yt)
                res[9] = float(y2) * 100 / float(yt)
            else:
                res[8] = md.tr_history_year[course][3]
                res[9] = md.tr_history_year[course][4]
            return res
    print("can not find trainer %s" % (name,))
    return res


def get_game_info(date, rcno):
    if date.weekday() == 5:
        file_date = date + datetime.timedelta(days=-2)
    if date.weekday() == 6:
        file_date = date + datetime.timedelta(days=-3)
    fname = '../txt/1/chulma/chulma_1_%d%02d%02d.txt' % (file_date.year, file_date.month, file_date.day)
    #print(fname)
    finput = open(fname)
    date_s = "%02d[.]%02d[.]%02d" % (date.year % 100, date.month, date.day)
    exp = "%s.*%d경주" % (date_s, rcno)
    #print("%s" % exp)
    found = False
    for _ in range(3000):
        line = finput.readline()
        if not line:
            break
        line = unicode(line, 'euc-kr').encode('utf-8')
        if re.search(unicode(r'%s' % exp, 'utf-8').encode('utf-8'), line) is not None:
            found = True
            break
    if not found:
        return [-1, -1]
    for _ in range(5):
        line = finput.readline()
        if not line:
            break
        line = unicode(line, 'euc-kr').encode('utf-8')
        #print("%s" % line)
        num = re.search(unicode(r'(?<=출전:)[\s\d]+(?=두)', 'utf-8').encode('utf-8'), line)
        kind = re.search(unicode(r'\d+(?=등급)', 'utf-8').encode('utf-8'), line)
        if num is not None:
            if kind is None:
                kind = 0
            else:
                kind = int(kind.group()[-1])
            return [num.group(), kind]
    return [-1, -1]


def get_race_record():
    race_record = RaceRecord()
    with gzip.open('../data/race_record.gz', 'rb') as f:
        tmp_dict = _pickle.loads(f.read())
        race_record.__dict__.update(tmp_dict)
    return race_record

def parse_xml_entry(meet, date_i, number, md3):
    # get other data
    data_hr = xh.parse_xml_hr(meet)
    data_jk = xj.parse_xml_jk(meet)
    data_tr = xt.parse_xml_tr(meet)
    date_m = date_i / 100
    date = datetime.date(date_i/10000, date_i/100%100, date_i%100)
    data = []
    filename = '../xml/entry/get_entry_%d_%d.xml' % (meet, date_m)
    file_input = open(filename)
    print("process in %s" % filename)
    response_body = file_input.read()
    xml_text = BeautifulSoup(response_body, 'html.parser')
    humidity = get_humidity()
    jangu_clinic = wc.parse_hr_clinic(date)
    race_record = get_race_record()
    mean_data, weight = gdd.make_mean_race_record(race_record)
    for itemElm in xml_text.findAll('item'):
        #print itemElm
        course = int(unicode(itemElm.rcdist.string))
        month = date_i/100%100
        rcdate = int("%s%s%s" % (itemElm.rcdate.string[:4], itemElm.rcdate.string[5:7], itemElm.rcdate.string[8:10]))
        rcno = int("%s" % (itemElm.rcno.string))
        if date_i != rcdate:
            continue
        if number > 0 and number != rcno:
            continue
        hr_gender, hr_days = get_hr_data(data_hr, itemElm.hrname.string)
        hr_weight = gdd.get_weight(meet, date_i, int(itemElm.rcno.string), itemElm.hrname.string, course)
        hr_dweight = gdd.get_dweight(meet, date_i, int(itemElm.rcno.string), itemElm.hrname.string)
        hr_dist_rec = gdd.get_distance_record(meet, itemElm.hrname.string, int(itemElm.rcno.string), date, course, md)
        cnt, kind = get_game_info(datetime.date(date_i / 10000, date_i / 100 % 100, date_i % 100), int(itemElm.rcno.string))
        hr_win = get_hr_win(itemElm.cntt.string, itemElm.ord1t.string, itemElm.ord2t.string, itemElm.cnty.string,
                           itemElm.ord1y.string, itemElm.ord2y.string, course)
        jk_win = get_jk_win(data_jk, itemElm.jkname.string, course)
        tr_win = get_tr_win(data_tr, itemElm.trname.string, course)
		
        hrname = unicode(itemElm.hrname.string).encode('utf-8')
        dbudam = gdd.get_dbudam(meet, date_i, int(rcno), hrname)
        drweight = gdd.get_drweight(meet, date_i, int(rcno), hrname)
        lastday = gdd.get_lastday(meet, date_i, int(rcno), hrname)
        train_state = gdd.get_train_state(meet, date_i, int(rcno), hrname)
        hr_no = gdd.get_hrno(meet, date_i, int(rcno), hrname)
        lineage_info = gl.get_lineage(1, hr_no, 'File')
        race_records, weight_past = gdd.get_hr_race_record(hrname, date_i, race_record, md3, mean_data, weight)
        if hr_weight == -1:
            hr_weight = weight_past
        if hr_weight == 0:
            hr_weight = {1000: 461, 1100: 460, 1200: 463, 1300: 464, 1400: 466, 1700: 466, 1800: 471, 1900: 475, 2000: 482, 2300: 492}[course]
        jc_data = wc.get_jangu_clinic(jangu_clinic, hrname)

        adata = [int(unicode(itemElm.rcdist.string)), int(humidity), int(kind), dbudam, drweight, lastday]
        assert len(adata) == 6
        adata.extend(train_state)
        assert len(adata) == 12
        adata.extend(lineage_info)
        assert len(adata) == 74
        adata.extend([int(itemElm.chulno.string), itemElm.hrname.string, itemElm.prdctyname.string, hr_gender,
                 int(itemElm.age.string), itemElm.wgbudam.string, itemElm.jkname.string, itemElm.trname.string,
                 itemElm.owname.string, hr_weight, hr_dweight, int(cnt), int(itemElm.rcno.string), date_i / 100 % 100,
                 hr_days])
        assert len(adata) == 89
        adata.extend(hr_win)
        assert len(adata) == 99
        adata.extend(hr_dist_rec)
        assert len(adata) == 105
        adata.extend(jk_win)
        assert len(adata) == 115
        adata.extend(tr_win)
        assert len(adata) == 125
        adata.extend(jc_data)
        assert len(adata) == 206
        adata.extend(race_records)
        assert len(adata) == 234
        data.append(adata)

    df = pd.DataFrame(data)
    df.columns = ['course', 'humidity', 'kind', 'dbudam', 'drweight', 'lastday', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'ts6'] \
                + ['lg%d'%i for i in range(1,63)] \
                + ['idx', 'name', 'cntry', 'gender', 'age', 'budam', 'jockey', 'trainer', 'owner', # 9
                  'weight', 'dweight', 'cnt', 'rcno', 'month', 'hr_days', 'hr_nt', 'hr_nt1', 'hr_nt2', 'hr_t1', 'hr_t2', 'hr_ny', 'hr_ny1', # 13
                  'hr_ny2', 'hr_y1', 'hr_y2', 'hr_dt', 'hr_d1', 'hr_d2', 'hr_rh', 'hr_rm', 'hr_rl', # 9
                  'jk_nt', 'jk_nt1', 'jk_nt2', 'jk_t1', 'jk_t2', 'jk_ny', 'jk_ny1', # 7
                  'jk_ny2', 'jk_y1', 'jk_y2', 'tr_nt', 'tr_nt1', 'tr_nt2', 'tr_t1', 'tr_t2', 'tr_ny', 'tr_ny1', # 10
                  'tr_ny2', 'tr_y1', 'tr_y2'] \
                  + ['jc%d'%i for i in range(1,82)] \
                  + ['rc%d'%i for i in range(1,29)]
    return df


# 73 - 4 = 59


if __name__ == '__main__':
    meet = 1
    rcno = 10
    date = 20161224
    data = parse_xml_entry(meet, date, rcno)
    data.to_csv('../log/xml_%d_%d.csv' % (date, rcno))
    print(data)

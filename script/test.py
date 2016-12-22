# -*- coding:utf-8 -*-
from __future__ import print_function

import re
import pandas as pd
from urllib2 import Request, urlopen
import random
import datetime
import os
from bs4 import BeautifulSoup
import numpy as np
from sklearn.externals import joblib
from mean_data import mean_data

def get_humidity():
    url = "http://race.kra.co.kr/chulmainfo/trackView.do?Act=02&Sub=10&meet=1"
    response_body = urlopen(url).read()
    line = unicode(response_body, 'euc-kr').encode('utf-8')
    print("%s" % line)
    p = re.compile(unicode(r'(?<=함수율 <span>: )\d+(?=\%\()', 'utf-8').encode('utf-8'), re.MULTILINE)
    pl = p.search(line)
    res = 10
    if pl is not None:
        res = pl.group()
    return res


def get_hr_weight(meet, date, rcno, hrname):
    #url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoWeight.do?Act=02&Sub=1&meet=1&rcNo=11&rcDate=20161030"
    url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoWeight.do?Act=02&Sub=1&meet=%s&rcNo=%s&rcDate=%s" % (meet, rcno, date)
    response_body = urlopen(url).read()
    line = unicode(response_body, 'euc-kr').encode('utf-8')

    #hrname = "%s" % hrname
    exp = '%s</a></td>\s+<td>\d+</td>\s+<td>[-\d]+(?=</td>)' % hrname
    p = re.compile(unicode(r'%s' % exp, 'utf-8').encode('utf-8'), re.MULTILINE)
    pl = p.search(line)
    res = 10
    if pl is not None:
        weight = pl.group().split('<td>')[1].split('</td>')[0]
        dweight = pl.group().split('<td>')[2]
        res = (weight, dweight)
    return res


def test():
    df = pd.DataFrame([[1,2,3],[4,5,6]])
    df.columns = ['a', 'b', 'c']
    for idx, rows in df.iterrows():
        print(rows['a'])


def df_concat():
    a = pd.DataFrame([[1,2,3,4,5], [2,3,4,5,6]])
    a.columns = ['a', 'b', 'c', 'd', 'e']
    b = pd.DataFrame([4,3])
    print(pd.concat([a[['c','a']], b], axis=1))


def test_random():
    total = 10
    print(random.randint(1, total))


def get_distance_record_url(hrname, rcno, date):
    print("name: %s, rcno: %d, date: %d" % (hrname, rcno, date))
    url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=1&rcNo=%d&rcDate=%d" % (rcno, date)
    response_body = urlopen(url).read()
    line = unicode(response_body, 'euc-kr').encode('utf-8')
    exp = '%s.+\s+.+\s+<td>\d+[.]\d+</td>\s+<td>\d+[.]\d+</td>\s+<td>\d+[:]\d+[.]\d+.+</td>\s+<td>\d+[:]\d+[.]\d+.+</td>\s+<td>\d+[:]\d+[.]\d+.+</td>' % hrname
    p = re.compile(r'%s' % exp, re.MULTILINE)
    pl = p.search(line)
    res = [-1, -1, -1, -1, -1, -1]
    if pl is not None:
        pls = pl.group().split()
        res[0] = re.search(unicode(r'\d+(?=\()', 'utf-8').encode('utf-8'), pls[2]).group()
        res[1] = re.search(unicode(r'\d+[.]\d+', 'utf-8').encode('utf-8'), pls[3]).group()
        res[2] = re.search(unicode(r'\d+[.]\d+', 'utf-8').encode('utf-8'), pls[4]).group()
        t = re.search(unicode(r'\d+[:]\d+[.]\d+', 'utf-8').encode('utf-8'), pls[5]).group()
        res[3] = int(t.split(':')[0])*600 + int(t.split(':')[1].split('.')[0])*10 + int(t.split('.')[1])
        t = re.search(unicode(r'\d+[:]\d+[.]\d+', 'utf-8').encode('utf-8'), pls[6]).group()
        res[4] = int(t.split(':')[0])*600 + int(t.split(':')[1].split('.')[0])*10 + int(t.split('.')[1])
        t = re.search(unicode(r'\d+[:]\d+[.]\d+', 'utf-8').encode('utf-8'), pls[7]).group()
        res[5] = int(t.split(':')[0])*600 + int(t.split(':')[1].split('.')[0])*10 + int(t.split('.')[1])
    else:
        print("can not find %s in %s" % (hrname, url))
    return res


def get_fname_dist(date, rcno):
    while True:
        date_s = int("%d%02d%02d" % (date.year, date.month, date.day))
        filename = '../txt/1/dist_rec/dist_rec_1_%s_%d.txt' % (date_s, rcno)
        if os.path.isfile(filename):
            return filename
        if date.weekday() == 5:
            date = date + datetime.timedelta(days=-6)
        elif date.weekday() == 6:
            date = date + datetime.timedelta(days=-1)
    return -1


def get_grade():
    str = "(서울) 제82일 1300M 국6    별정A       경주명 : 일반 "
    kind = re.search(unicode(r'M.+\d', 'utf-8').encode('utf-8'), str)
    if kind is None:
        print("none")
    else:
        print(kind.group()[-1])


def get_game_info(date, rcno):
    if date.weekday() == 5:
        file_date = date + datetime.timedelta(days=-2)
    if date.weekday() == 6:
        file_date = date + datetime.timedelta(days=-3)
    fname = '../txt/1/chulma/chulma_1_%d%02d%02d.txt' % (file_date.year, file_date.month, file_date.day)
    print(fname)
    finput = open(fname)
    date_s = "%d[.]%02d[.]%02d" % (date.year % 100, date.month, date.day)
    exp = "%s.*%d" % (date_s, rcno)
    print("%s" % exp)
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
        print("%s" % line)
        num = re.search(unicode(r'(?<=출전:)[\s\d]+(?=두)', 'utf-8').encode('utf-8'), line)
        kind = re.search(unicode(r'\d+(?=등급)', 'utf-8').encode('utf-8'), line)
        if num is not None:
            return [num.group(), kind.group()[-1]]
    return [-1, -1]


def pandas_compare():
    df = pd.DataFrame([['가',2,3,4,5], ['나',3,4,5,6], ['다',4,5,6,7]], columns=['A', 'B', 'C', 'D', 'E'])
    df.loc[df['A'] == '가', 'A'] = 4
    print(df)


def get_rate():
    line = "     복연: ⑨⑤3.9 ⑨⑦16.0 ⑤⑦110.4"
    line = "     복연:④⑬   2.1   ④⑧   7.2   ⑬⑧   6.0"
    line = "배당률 단: ⑨5.5        연: ⑨1.7 ⑧1.9 ③3.6          복: ⑨⑧11.7      4F:50.0"
    num_circle_list = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭"  #
    bokyeon = [-1, -1, -1]
    boksik = ''
    ssang = ''
    sambok = ''
    res = re.search(r'(?<= 복:).+(?=4F)', line)
    if res is not None:
        res = res.group().split()
        if len(res) == 2:
            boksik.append("%s%s" % (res[0], res[1]))
        elif len(res) == 1:
            boksik = res[0]
        else:
            print("not expected.. %s" % line)
    res = re.search(r'(?<= 쌍:).+', line)
    if res is not None:
        res = res.group().split()
        if len(res) == 2:
            ssang.append("%s%s" % (res[0], res[1]))
        elif len(res) == 1:
            ssang = res[0]
        else:
            print("not expected.. %s" % line)
    res = re.search(r'(?<=복연:).+', line)
    if res is not None:
        res = res.group().split()
        if len(res) == 6:
            bokyeon.append("%s%s" % (res[0], res[1]))
            bokyeon.append("%s%s" % (res[2], res[3]))
            bokyeon.append("%s%s" % (res[4], res[5]))
        elif len(res) == 3:
            bokyeon = res[0]
        else:
            print("not expected.. %s" % line)
    res = re.search(r'(?<=삼복:).+', line)
    if res is not None:
        res = res.group().split()
        if len(res) == 2:
            sambok.append("%s%s" % (res[0], res[1]))
        elif len(res) == 1:
            sambok = res[0]
        else:
            print("not expected.. %s" % line)
    print("%s, %s, %s" % (bokyeon[0], bokyeon[1], bokyeon[2]))
    print("복: %s, 쌍: %s, 삼복: %s" % (boksik, ssang, sambok))


def get_dbudam(meet, date, rcno, name):
    fname = '../txt/%d/chulmapyo/chulmapyo_%d_%d_%d.txt' % (meet, meet, date, rcno)
    if os.path.exists(fname):
        response_body = open(fname).read()
    else:
        base_url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoChulmapyo.do?Act=02&Sub=1&"
        url = base_url + "meet=%d&rcNo=%d&rcDate=%d" % (meet, rcno, date)
        response_body = urlopen(url).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr'):
            itemList = itemElm2.findAll('td')
            if name == itemList[1].string.encode('utf-8'):
                return itemList[7].string
    return -1



def get_budam(meet, date, rcno, name):
    fname = '../txt/%d/chulmapyo/chulmapyo_%d_%d_%d.txt' % (meet, meet, date, rcno)
    if os.path.exists(fname):
        response_body = open(fname).read()
    else:
        base_url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoChulmapyo.do?Act=02&Sub=1&"
        url = base_url + "meet=%d&rcNo=%d&rcDate=%d" % (meet, rcno, date)
        response_body = urlopen(url).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr'):
            itemList = itemElm2.findAll('td')
            if name == itemList[1].string.encode('utf-8'):
                return itemList[6].string
    return -1


def get_weight(meet, date, rcno, name):
    fname = '../txt/%d/weight/weight_%d_%d_%d.txt' % (meet, meet, date, rcno)
    if os.path.exists(fname):
        response_body = open(fname).read()
    else:
        base_url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoWeight.do?Act=02&Sub=1&"
        url = base_url + "meet=%d&rcNo=%d&rcDate=%d" % (meet, rcno, date)
        response_body = urlopen(url).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr'):
            itemList = itemElm2.findAll('td')
            print(itemList)
            if name == itemList[1].string.encode('utf-8'):
                return itemList[2].string
    return -1


def get_dweight(meet, date, rcno, name):
    fname = '../txt/%d/weight/weight_%d_%d_%d.txt' % (meet, meet, date, rcno)
    if os.path.exists(fname):
        response_body = open(fname).read()
    else:
        base_url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoWeight.do?Act=02&Sub=1&"
        url = base_url + "meet=%d&rcNo=%d&rcDate=%d" % (meet, rcno, date)
        response_body = urlopen(url).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr'):
            itemList = itemElm2.findAll('td')
            if name == itemList[1].string.encode('utf-8'):
                return itemList[3].string
    return -1


def get_drweight(meet, date, rcno, name):
    fname = '../txt/%d/weight/weight_%d_%d_%d.txt' % (meet, meet, date, rcno)
    if os.path.exists(fname):
        response_body = open(fname).read()
    else:
        base_url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoWeight.do?Act=02&Sub=1&"
        url = base_url + "meet=%d&rcNo=%d&rcDate=%d" % (meet, rcno, date)
        response_body = urlopen(url).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr'):
            itemList = itemElm2.findAll('td')
            if name == itemList[1].string.encode('utf-8'):
                last_date = itemList[4].string
                last_date = datetime.date(int(last_date[:4]), int(last_date[5:7]), int(last_date[8:10]))
                delta_day = datetime.date(date/10000, date/100%100, date%100) - last_date
                return float(itemList[3].string) / delta_day.days
    return -1


def load_save_csv():
    df = pd.read_csv('../log/2016.csv')
    df.to_csv('../log/2016_2.csv', index=False)


def get_num(line):
    num_circle_list = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭"
    a = num_circle_list.find(line[:3]) / 3
    b = num_circle_list.find(line[3:6]) / 3
    r = float(line[6:])
    return [a, b, r]



def typecheck(input):
    print(type(input))
    print(type(1))
    print(type(1.1))
    if re.search(r'\d', input[:1]) is None:
        print("it's not a number")
    else:
        print("it's a number")


def make_dir(cmd):
    os.system(cmd)


def make_df():
    df = pd.DataFrame([[1,2,3,20161119],[2,3,4,20161120]])
    df.columns = ['a','b','c','date']
    df.to_csv('test.csv', index=False)


def update_df():
    df = pd.read_csv('../data/1_2007_2016.csv')
    print(df.loc[len(df)-1]['date'])


def get_distance_record(meet, name, rcno, date, course):
    name = name.replace('★', '')
    date_i = int("%d%02d%02d" % (date.year, date.month, date.day))
    fname = '../txt/%d/dist_rec/dist_rec_%d_%d_%d.txt' % (meet, meet, date_i, rcno)
    res = []
    cand = "조보후승기"
    if os.path.exists(fname):
        response_body = open(fname).read()
    else:
        base_url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&"
        url = base_url + "meet=%d&rcNo=%d&rcDate=%d" % (meet, rcno, date_i)
        response_body = urlopen(url).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr'):
            itemList = itemElm2.findAll('td')
            if name in itemList[1].string.encode('utf-8'):
                if int(unicode(itemList[2].string)[0]) == 0:
                    if int(course) == 1000:
                        return [0, 0, 0, 636, 642, 648]
                    elif int(course) == 1100:
                        return [0, 0, 0, 702, 705, 708]
                    elif int(course) == 1200:
                        return [0, 0, 0, 771, 779, 789]
                    elif int(course) == 1300:
                        return [0, 0, 0, 837, 845, 853]
                    elif int(course) == 1400:
                        return [0, 0, 0, 894, 904, 915]
                    elif int(course) == 1700:
                        return [0, 0, 0, 1133, 1143, 1154]
                    elif int(course) == 1800:
                        return [0, 0, 14.3, 1193, 1206, 1221]
                    elif int(course) == 1900:
                        return [0, 0, 9.1, 1256, 1270, 1283]
                    elif int(course) == 2000:
                        return [0, 0, 16.7, 1308, 1329, 1347]
                    elif int(course) == 2300:
                        return [0, 0, 16.7, 1497, 1512, 1529]
                res.append(int(unicode(itemList[2].string)[0]))
                res.append(float(unicode(itemList[3].string)))
                res.append(float(unicode(itemList[4].string)))
                t = unicode(itemList[5].string)
                res.append(int(t.split(':')[0]) * 600 + int(t.split(':')[1].split('.')[0]) * 10 + int(t.split('.')[1][0]))
                t = unicode(itemList[6].string)
                res.append(int(t.split(':')[0]) * 600 + int(t.split(':')[1].split('.')[0]) * 10 + int(t.split('.')[1][0]))
                t = unicode(itemList[7].string)
                res.append(int(t.split(':')[0]) * 600 + int(t.split(':')[1].split('.')[0]) * 10 + int(t.split('.')[1][0]))
    if len(res) == 6:
        return res
    else:
        print("can not find %s in %s" % (name, fname))
        if int(course) == 1000:
            return [2, 0, 0, 636, 642, 648]
        elif int(course) == 1100:
            return [1, 0, 0, 702, 705, 708]
        elif int(course) == 1200:
            return [3, 0, 0, 771, 779, 789]
        elif int(course) == 1300:
            return [2, 0, 0, 837, 845, 853]
        elif int(course) == 1400:
            return [3, 0, 0, 894, 904, 915]
        elif int(course) == 1700:
            return [2, 0, 0, 1133, 1143, 1154]
        elif int(course) == 1800:
            return [3, 0, 14.3, 1193, 1206, 1221]
        elif int(course) == 1900:
            return [3, 0, 9.1, 1256, 1270, 1283]
        elif int(course) == 2000:
            return [4, 0, 16.7, 1308, 1329, 1347]
        elif int(course) == 2300:
            return [2, 0, 16.7, 1497, 1512, 1529]


def get_hrno(meet, date, rcno, name):
    fname = '../txt/%d/chulmapyo/chulmapyo_%d_%d_%d.txt' % (meet, meet, date, rcno)
    if os.path.exists(fname):
        response_body = open(fname).read()
    else:
        base_url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoChulmapyo.do?Act=02&Sub=1&"
        url = base_url + "meet=%d&rcNo=%d&rcDate=%d" % (meet, rcno, date)
        response_body = urlopen(url).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr'):
            itemList = itemElm2.findAll('td')
            if name == itemList[1].string.encode('utf-8'):
                return re.search(r'\d{6}', unicode(itemList[1])).group()
    return -1



def get_hr_testrecord(meet, hrno):
    base_url = "http://race.kra.co.kr/racehorse/profileTrainCheck.do?Act=02&Sub=1&"
    url = base_url + "meet=%d&hrNo=%06d" % (meet, hrno)
    response_body = urlopen(url).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for itemElm in xml_text.findAll('tbody')[1:]:
        for itemElm2 in itemElm.findAll('tr'):
            itemList = itemElm2.findAll('td')
            if '주행' == unicode(itemList[2].string).encode('utf-8'):
                print(itemList)
    return -1


def norm_racescore(meet, course, humidity, value):
    div_data = {1000: [0.985, 1.003, 1.003, 1.001, 0.999, 1.002, 0.999, 1.002, 1.002, 1.001, 1.003, 1.002, 1.001, 0.995, 1.004, 0.999, 0.998, 0.994, 0.995, 0.992],
                1100: [0.979, 1.015, 0.999, 1.002, 0.996, 0.997, 1.004, 1.012, 0.999, 1.009, 1.003, 1.000, 1.000, 0.999, 1.006, 0.997, 0.994, 0.995, 0.991, 0.993],
                1200: [0.991, 0.995, 1.004, 1.001, 1.002, 1.000, 0.997, 0.999, 1.004, 1.004, 1.005, 1.001, 1.003, 0.999, 1.002, 0.993, 0.996, 0.995, 0.996, 0.992],
                1300: [0.993, 0.992, 1.005, 1.001, 0.999, 1.000, 0.998, 1.001, 1.005, 1.006, 1.002, 1.003, 1.000, 0.996, 1.003, 0.992, 0.997, 0.998, 0.997, 0.991],
                1400: [0.993, 0.995, 1.003, 1.001, 1.002, 1.001, 1.000, 1.000, 1.005, 1.007, 1.005, 1.000, 0.997, 1.001, 0.996, 0.993, 0.997, 0.993, 0.995, 0.990],
                1700: [0.978, 0.995, 1.003, 1.001, 1.002, 1.002, 1.000, 1.002, 1.001, 1.005, 1.006, 1.002, 1.001, 1.002, 0.992, 0.997, 0.994, 0.990, 0.989, 0.992],
                1800: [0.984, 0.993, 1.002, 1.001, 1.002, 1.001, 1.000, 1.002, 1.004, 1.004, 1.003, 1.002, 1.004, 1.002, 0.995, 0.998, 0.997, 0.995, 0.991, 0.989],
                1900: [0.979, 0.997, 1.005, 1.001, 1.002, 1.001, 1.000, 1.002, 1.006, 1.013, 1.007, 1.003, 1.001, 1.001, 0.987, 0.995, 0.985, 0.995, 0.989, 0.986],
                2000: [0.979, 0.997, 1.004, 1.001, 1.000, 1.001, 1.002, 1.002, 1.007, 1.006, 1.003, 1.008, 1.015, 0.982, 0.988, 0.998, 0.992, 0.991, 0.983, 0.993],
                2300: [0.979, 0.995, 1.009, 1.016, 1.024, 0.999, 1.016, 1.003, 1.003, 0.996, 0.985, 1.000, 0.997, 0.997, 0.999, 0.995, 0.987, 0.995, 0.983, 0.984]}
    if humidity >= 20:
        humidity = 20
    return (value / div_data[course][humidity-1])


def get_hr_racescore(meet, hrno, _date):
    result = [-1, -1, -1, -1, -1, -1] # 주, 1000, 1200, 1300, 1400, 1700
    race_sum = [[], [], [], [], [], []]
    fname = '../txt/%d/racescore/racescore_%d_%06d.txt' % (meet, meet, hrno)
    if os.path.exists(fname):
        response_body = open(fname).read()
    else:
        base_url = "http://race.kra.co.kr/racehorse/profileRaceScore.do?Act=02&Sub=1&"
        url = base_url + "meet=%d&hrNo=%06d" % (meet, hrno)
        response_body = urlopen(url).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr')[2:]:
            itemList = itemElm2.findAll('td')
            #print(itemList)
            date = re.search(r'\d{4}/\d{2}/\d{2}', unicode(itemList[1])).group()
            date = int("%s%s%s" % (date[:4], date[5:7], date[8:]))
            if date > _date:
                continue
            racekind = unicode(itemList[3].string).strip().encode('utf-8')
            try:
                distance = int(unicode(itemList[4].string).strip().encode('utf-8'))
            except:
                distance = 1000
            record = unicode(itemList[10].string).strip().encode('utf-8')
            #print(unicode(itemList[12].string).strip())
            humidity = int(re.search(r'\d+', unicode(itemList[12].string)).group())
            try:
                record = int(record[0])*600 + int(record[2:4])*10 + int(record[5])
            except:
                record = -1
            #print("주, 일, %s" % racekind)
            record = norm_racescore(1, distance, humidity, record)
            if racekind == '주':
                race_sum[0].append(record)
            elif racekind == '일':
                if distance == 1000:
                    race_sum[1].append(record)
                elif distance == 1200:
                    race_sum[2].append(record)
                elif distance == 1300:
                    race_sum[3].append(record)
                elif distance == 1400:
                    race_sum[4].append(record)
                elif distance == 1700:
                    race_sum[5].append(record)
            print("%d, %s, %s, %d" % (date, racekind, distance, record))
    for i in range(len(race_sum)):
        if len(race_sum[i]) == 0:
            result[i] = -1
        else:
            result[i] = np.mean(race_sum[i])
    return result
    # hrno = get_hrno(1, 20161106, 11, "구만석")
    # print(get_hr_racescore(1,int(hrno), 20161106))


def dump_data(data, fname):
    joblib.dump(data, fname)


def simple_list_concat():
    return [0] + [1,2,3,4][2:]


def update_md(fname):
    data = pd.read_csv(fname)
    md = mean_data()
    md.update_data(data)
    md = joblib.load(fname)
    md = joblib.load(fname)


def check_average_mean(fname, md=mean_data()):
    data = pd.read_csv(fname)
    prev_date = data['date'][0]
    for idx, row in data.iterrows():
        course = int(row['course'])
        humidity = int(row['humidity'])
        if course not in [1000, 1200, 1300, 1400, 1700]:
            continue
        try:
            #md.update_race_score(course, humidity, row)
            if prev_date != row['date']:
                for i in range(21):
                    print("%.4f" % md.race_score[1000][i], end=' ')
                print()
                prev_date = row['date']
            humidity = min(humidity, 20) - 1
            record = row['rctime'] / md.race_score[course][humidity] * md.race_score[course][20]
            #if record < md.race_score[course][20]*0.8 or record > md.race_score[course][20]*1.2:
            #    continue
            md.race_score[course][humidity] += 0.1 * (row['rctime'] - md.race_score[course][humidity])
            md.race_score[course][20] = np.mean(md.race_score[course])
            #print("value: %f, %f, %f, %f, %f" % (md.race_score[1000][20], md.race_score[1200][20], md.race_score[1300][20], md.race_score[1400][20], md.race_score[1700][20]))
            #print("value: %f, %f, %f, %f, %f" % (md.race_score[1000][20], md.race_score[1200][20], md.race_score[1300][20], md.race_score[1400][20], md.race_score[1700][20]))
            #print("value: %f" % (md.race_score[course][20]))
        except:
            print("humidity: %d, racetime: %f" % (humidity, row['rctime']))


def check_md():
    md = joblib.load('../data/1_2007_2016_v1.9_md_2016-12-10.pkl')
    print(md.race_score[900][20])
    print(md.race_score[1000][20])
    print(md.race_score[1200][20])
    print(md.race_score[1300][20])
    print(md.race_score[1400][20])
    print(md.race_score[1700][20])


def print_all():
    cand = [[5], [3,16,1,13,7,12], [3,16,1,13,7,12]]
    cnt = 0
    for x in cand[0]:
        for y in cand[1]:
            for z in cand[2]:
                if x == y or x == z or y == z:
                    continue
                cnt += 1
    bet = 100 / cnt
    print("bet: %f" % bet)



if __name__ == '__main__':
    print_all()
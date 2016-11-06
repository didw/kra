# -*- coding:utf-8 -*-

import re
import pandas as pd
from urllib2 import Request, urlopen
import random
import datetime
import os


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

def test_rank():
    data = pd.Series([1,4,3,2])
    top = data.rank()
    print top[0] in [1, 2]
    print top[1] in [1, 2]
    print top[2] in [1, 2]
    print top[3] in [1, 2]

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
    print(line)
    exp = '%s.+\s+.+\s+<td>\d+[.]\d+</td>\s+<td>\d+[.]\d+</td>\s+<td>\d+[:]\d+[.]\d+.+</td>\s+<td>\d+[:]\d+[.]\d+.+</td>\s+<td>\d+[:]\d+[.]\d+.+</td>' % hrname
    p = re.compile(unicode(r'%s' % exp, 'utf-8').encode('utf-8'), re.MULTILINE)
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

def get_distance_record(hrname, rcno, date):
    filename = get_fname_dist(date, rcno)
    f_input = open(filename)
    res = []
    found = False
    for line in f_input:
        if not line:
            break
        line = unicode(line, 'euc-kr').encode('utf-8')
        if re.search(unicode(r'%s' % hrname, 'utf-8').encode('utf-8'), line) is not None:
            found = True
            break
        if not found:
            continue

    for line in f_input:
        line = unicode(line, 'euc-kr').encode('utf-8')
        if not line or len(res) == 6:
            break
        dnt = re.search(unicode(r'(?<=>)\d+(?=\()', 'utf-8').encode('utf-8'), line)
        if dnt is not None:
            if dnt.group() == '0':
                return [-1, -1, -1, -1, -1, -1]
            res.append(dnt.group())
            continue
        dn = re.search(unicode(r'(?<=<td>)\d+[.]\d+(?=</td>)', 'utf-8').encode('utf-8'), line)
        if dn is not None:
            res.append(dn.group())
            continue
        dr = re.search(unicode(r'(?<=<td>)\d+[:]\d+[.]\d+', 'utf-8').encode('utf-8'), line)
        if dr is not None:
            t = dr.group()
            res.append(int(t.split(':')[0])*600 + int(t.split(':')[1].split('.')[0])*10 + int(t.split('.')[1]))
            continue
    if len(res) == 6:
        return res
    else:
        print("can not find %s in %s" % (hrname, filename))
        return [-1, -1, -1, -1, -1, -1]


def get_grade():
    str = "(서울) 제82일 1300M 국6    별정A       경주명 : 일반 "
    kind = re.search(unicode(r'M.+\d', 'utf-8').encode('utf-8'), str)
    if kind is None:
        print("none")
    else:
        print(kind.group()[-1])


if __name__ == '__main__':
    print(get_distance_record('하이택시', 1, datetime.date(2007, 11, 3)))


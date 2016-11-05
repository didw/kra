# -*- coding:utf-8 -*-

import re
import pandas as pd
from urllib2 import Request, urlopen
import random


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


def get_distance_record(hrname, date):
    url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=1&rcNo=5&rcDate=%d" % date
    response_body = urlopen(url).read()
    line = unicode(response_body, 'euc-kr').encode('utf-8')
    print("%s" % line)
    #exp = '%s[\w\s]+<td>\d+[.]\d+</td>\s+<td>\d+[.]\d+</td>\s+<td>\d+[:]\d+[.]\d+\w</td>\s+<td>\d+[:]\d+[.]\d+\w</td>\s+<td>\d+[:]\d+[.]\d+\w</td>' % hrname
    exp = '%s.+\s+.+\s+<td>\d+[.]\d+</td>\s+<td>\d+[.]\d+</td>\s+<td>\d+[:]\d+[.]\d+.+</td>\s+<td>\d+[:]\d+[.]\d+.+</td>\s+<td>\d+[:]\d+[.]\d+.+</td>' % hrname
    p = re.compile(unicode(r'%s' % exp, 'utf-8').encode('utf-8'), re.MULTILINE)
    pl = p.search(line)
    res = 10
    if pl is not None:
        pls = pl.group().split()
        res[0] = re.compile(unicode(r'\d+(?=\()', 'utf-8').encode('utf-8'), pls[2]).group()
        res[1] = re.compile(unicode(r'\d+[.]\d+', 'utf-8').encode('utf-8'), pls[3]).group()
        res[2] = re.compile(unicode(r'\d+[.]\d+', 'utf-8').encode('utf-8'), pls[4]).group()
        res[3] = re.compile(unicode(r'\d+[:]\d+[.]\d+', 'utf-8').encode('utf-8'), pls[5]).group()
        res[4] = re.compile(unicode(r'\d+[:]\d+[.]\d+', 'utf-8').encode('utf-8'), pls[6]).group()
        res[5] = re.compile(unicode(r'\d+[:]\d+[.]\d+', 'utf-8').encode('utf-8'), pls[7]).group()
    else:
        print("None")
    return res


if __name__ == '__main__':
    print(get_distance_record("야호챔피언", 20161105))


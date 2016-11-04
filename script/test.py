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
    data = pd.Series([1,4,2,3])
    top = data.rank()
    print(data)
    print(top)
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


if __name__ == '__main__':
    test_rank()


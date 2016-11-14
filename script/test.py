# -*- coding:utf-8 -*-

import re
import pandas as pd
from urllib2 import Request, urlopen
import random
import datetime
import os
from bs4 import BeautifulSoup


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

if __name__ == '__main__':
    print(get_drweight(1, 20091212, 10, "풀스텝"))


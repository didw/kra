# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob
import re
from urllib2 import urlopen
import parse_xml_jk as xj
import parse_xml_hr as xh
import parse_xml_tr as xt
import parse_xml_train as xtr
import datetime
import sys
reload(sys)
sys.setdefaultencoding('utf-8')




def get_humidity():
    url = "http://race.kra.co.kr/chulmainfo/trackView.do?Act=02&Sub=10&meet=1"
    response_body = urlopen(url).read()
    line = unicode(response_body, 'euc-kr').encode('utf-8')
    #print("%s" % line)
    p = re.compile(unicode(r'(?<=함수율 <span>: )\d+(?=\%\()', 'utf-8').encode('utf-8'), re.MULTILINE)
    pl = p.search(line)
    res = 10
    if pl is not None:
        res = pl.group()
    return res


def get_hr_weight(meet, date, rcno, hrname):
    #url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoWeight.do?Act=02&Sub=1&meet=1&rcNo=11&rcDate=20161030"
    date = "%s%s%s" % (date[:4], date[5:7], date[8:10])
    url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoWeight.do?Act=02&Sub=1&meet=%s&rcNo=%s&rcDate=%s" % (meet, rcno, date)
    response_body = urlopen(url).read()
    line = unicode(response_body, 'euc-kr').encode('utf-8')

    exp = '%s</a></td>\s+<td>\d+</td>\s+<td>[-\d]+(?=</td>)' % hrname
    p = re.compile(exp.encode('utf-8'), re.MULTILINE)
    pl = p.search(line)
    res = [-1, -1]
    if pl is not None:
        weight = pl.group().split('<td>')[1].split('</td>')[0]
        dweight = pl.group().split('<td>')[2]
        res = [weight, dweight]
    if res[0] == -1:
        print("Can not parsing weight %s in %s" % (hrname, url))
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
    print("can not find horse %s" % (name,))
    return (-1, -1)


def get_hr_win(tt, t1, t2, yt, y1, y2):
    res = [-1, -1, -1, -1]
    if int(tt) != 0:
        res[0] = int(t1) * 100 / int(tt)
        res[1] = int(t2) * 100 / int(tt)
    if int(yt) != 0:
        res[2] = int(y1) * 100 / int(yt)
        res[3] = int(y2) * 100 / int(yt)
    return res


def get_jk_win(data, name):
    res = [-1, -1, -1, -1]
    for idx, line in data.iterrows():
        if line['jkName'] == name:
            tt = int(line['cntT'])
            t1 = int(line['ord1T'])
            t2 = int(line['ord2T'])
            yt = int(line['cntY'])
            y1 = int(line['ord1Y'])
            y2 = int(line['ord2Y'])
            if int(tt) != 0:
                res[0] = int(t1) * 100 / int(tt)
                res[1] = int(t2) * 100 / int(tt)
            if int(yt) != 0:
                res[2] = int(y1) * 100 / int(yt)
                res[3] = int(y2) * 100 / int(yt)
            return res
    print("can not find jockey %s" % (name,))
    return res


def get_tr_win(data, name):
    res = [-1, -1, -1, -1]
    for idx, line in data.iterrows():
        if line['trName'] == name:
            tt = int(line['cntT'])
            t1 = int(line['ord1T'])
            t2 = int(line['ord2T'])
            yt = int(line['cntY'])
            y1 = int(line['ord1Y'])
            y2 = int(line['ord2Y'])
            if int(tt) != 0:
                res[0] = int(t1) * 100 / int(tt)
                res[1] = int(t2) * 100 / int(tt)
            if int(yt) != 0:
                res[2] = int(y1) * 100 / int(yt)
                res[3] = int(y2) * 100 / int(yt)
            return res
    print("can not find trainer %s" % (name,))
    return res


def parse_xml_entry(meet, date):
    # get other data
    data_hr = xh.parse_xml_hr(meet)
    data_jk = xj.parse_xml_jk(meet)
    data_tr = xt.parse_xml_tr(meet)
    data_train = xtr.parse_xml_train(date, meet)
    data = []
    filename = '../xml/entry/get_entry_%s_%s.xml' % (meet, date)
    file_input = open(filename)
    print "process in %s" % filename
    response_body = file_input.read()
    xml_text = BeautifulSoup(response_body, 'html.parser')
    humidity = get_humidity()
    for itemElm in xml_text.findAll('item'):
        #print itemElm
        """['course', 'humidity', 'rank', 'idx', 'name', 'cntry', 'gender', 'age', 'budam', 'jockey', \
          'trainer', 'owner', 'weight', 'dweight', 'rctime', 'r1', 'r2', 'r3', 'hr_days', 'hr_t1', \
          'hr_t2', 'hr_y1', 'hr_y2', 'jk_t1', 'jk_t2', 'jk_y1', 'jk_y2', 'tr_t1', 'tr_t2', 'tr_y1', \
          'tr_y2']
        """
        rcdate = datetime.date(int(itemElm.rcdate.string[:4]), int(itemElm.rcdate.string[5:7]), int(itemElm.rcdate.string[8:10]))
        if datetime.date.today() < rcdate:
            continue
        hr_gender, hr_days = get_hr_data(data_hr, itemElm.hrname.string)
        hr_weight, hr_dweight = get_hr_weight(meet, itemElm.rcdate.string, itemElm.rcno.string, itemElm.hrname.string)
        hr_win = get_hr_win(itemElm.cntt.string, itemElm.ord1t.string, itemElm.ord2t.string, itemElm.cnty.string,
                           itemElm.ord1y.string, itemElm.ord2y.string)
        jk_win = get_jk_win(data_jk, itemElm.jkname.string)
        tr_win = get_tr_win(data_tr, itemElm.trname.string)
        adata = [itemElm.rcdist.string,
                     humidity,
                     itemElm.chulno.string,
                     itemElm.hrname.string,
                     itemElm.prdctyname.string,
                     hr_gender,
                     itemElm.age.string,
                     itemElm.wgbudam.string,
                     itemElm.jkname.string,

                     itemElm.trname.string,
                     itemElm.owname.string,
                     hr_weight,
                     hr_dweight,
                     hr_days,
                     hr_win[0],
                     hr_win[1],
                     hr_win[2],
                     hr_win[3],

                     jk_win[0],
                     jk_win[1],
                     jk_win[2],
                     jk_win[3],

                     tr_win[0],
                     tr_win[1],
                     tr_win[2],
                     tr_win[3]]
        print(adata)
        data.extend(adata)

    df = pd.DataFrame(data)
    df.columns = ['course', 'humidity', 'idx', 'name', 'cntry', 'gender', 'age', 'budam', 'jockey', \
                  'trainer', 'owner', 'weight', 'dweight', 'rctime', 'r1', 'r2', 'r3', 'hr_days', 'hr_t1', \
                  'hr_t2', 'hr_y1', 'hr_y2', 'jk_t1', 'jk_t2', 'jk_y1', 'jk_y2', 'tr_t1', 'tr_t2', 'tr_y1', \
                  'tr_y2']
    return df


if __name__ == '__main__':
    meet = 1
    date = 201610
    data = parse_xml_entry(meet, date)
    print data

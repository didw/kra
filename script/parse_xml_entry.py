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
import os
import get_detail_data as gdd

reload(sys)
sys.setdefaultencoding('utf-8')


DEBUG = False

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
    hrname = hrname.replace('★'.encode('utf-8'), '')
    #url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoWeight.do?Act=02&Sub=1&meet=1&rcNo=11&rcDate=20161030"
    date = "%s%s%s" % (date[:4], date[5:7], date[8:10])
    url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoWeight.do?Act=02&Sub=1&meet=%s&rcNo=%s&rcDate=%s" % (meet, rcno, date)
    try:
        response_body = urlopen(url).read()
    except:
        print "can not read %s" % url
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
    res = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for idx, line in data.iterrows():
        if line['jkName'] == name:
            res[0] = tt = int(line['cntT'])
            res[1] = t1 = int(line['ord1T'])
            res[2] = t2 = int(line['ord2T'])
            res[5] = yt = int(line['cntY'])
            res[6] = y1 = int(line['ord1Y'])
            res[7] = y2 = int(line['ord2Y'])
            if int(tt) != 0:
                res[3] = int(t1) * 100 / int(tt)
                res[4] = int(t2) * 100 / int(tt)
            if int(yt) != 0:
                res[8] = int(y1) * 100 / int(yt)
                res[9] = int(y2) * 100 / int(yt)
            return res
    print("can not find jockey %s" % (name,))
    return res


def get_tr_win(data, name):
    res = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for idx, line in data.iterrows():
        if line['trName'] == name:
            res[0] = tt = int(line['cntT'])
            res[1] = t1 = int(line['ord1T'])
            res[2] = t2 = int(line['ord2T'])
            res[5] = yt = int(line['cntY'])
            res[6] = y1 = int(line['ord1Y'])
            res[7] = y2 = int(line['ord2Y'])
            if int(tt) != 0:
                res[3] = int(t1) * 100 / int(tt)
                res[4] = int(t2) * 100 / int(tt)
            if int(yt) != 0:
                res[8] = int(y1) * 100 / int(yt)
                res[9] = int(y2) * 100 / int(yt)
            return res
    print("can not find trainer %s" % (name,))
    return res


def get_distance_record_url(hrname, rcno, date):
    #print("name: %s, rcno: %d, date: %d" % (hrname, rcno, date))
    url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=2&rcNo=%d&rcDate=%d" % (rcno, int("%d%d%d" % (date.year, date.month, date.day)))
    response_body = urlopen(url).read()
    line = unicode(response_body, 'euc-kr').encode('utf-8')
    #print("%s" % line)
    exp = '%s.+\s+.+\s+<td>\d+[.]\d+</td>\s+<td>\d+[.]\d+</td>\s+<td>\d+[:]\d+[.]\d+.+</td>\s+<td>\d+[:]\d+[.]\d+.+</td>\s+<td>\d+[:]\d+[.]\d+.+</td>' % hrname.encode("utf-8")
    #print("exp: %s" % exp)
    pl = re.search(r'%s'%exp, line)
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

def get_game_info(date, rcno):
    if date.weekday() == 4:
        file_date = date + datetime.timedelta(days=-2)
    if date.weekday() == 5:
        file_date = date + datetime.timedelta(days=-3)
    fname = '../txt/2/chulma/chulma_2_%d%02d%02d.txt' % (file_date.year, file_date.month, file_date.day)
    #print("culma filename: %s" % fname)
    finput = open(fname)
    date_s = "%d[.]%02d[.]%02d" % (date.year % 100, date.month, date.day)
    exp = "%s.*%d" % (date_s, rcno)
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
        if DEBUG: print("%s" % line)
        num = re.search(unicode(r'(?<=출전:)[\s\d]+(?=두)', 'utf-8').encode('utf-8'), line)
        kind = re.search(unicode(r'(?<=두).+?\d', 'utf-8').encode('utf-8'), line)
        if num is not None:
            if kind is None:
                kind = 0
            else:
                if DEBUG: print("%s" % kind.group())
                kind = kind.group()[-1]
            return [num.group(), kind]
    return [-1, -1]



def get_fname(date, job):
    while True:
        date = date + datetime.timedelta(days=-1)
        date_s = int("%d%02d%02d" % (date.year, date.month, date.day))
        filename = '../txt/2/%s/%s_2_%s.txt' % (job, job, date_s)
        if os.path.isfile(filename):
            return filename
    return -1


# 이름             산지  성별   birth  -    조교사  마주명             -                    -                     총경기, 총1, 총2, 1년경기, 1년1, 1년2,총상금
# 킹메신저          한    수2014/03/08 2국6 18박대흥죽마조합            시에로골드          난초                    1    0    0    1    0    0    3000000                     0
def parse_txt_horse(date, rcno, name):
    filename = get_fname(date, "horse")
    f_input = open(filename)
    while True:
        line = f_input.readline()
        if not line:
            break
        line = unicode(line, 'euc-kr').encode('utf-8')
        if re.search(name.encode('utf-8'), line) is not None:
            data = []
            birth = re.search(unicode(r'\d{4}/\d{2}/\d{2}', 'utf-8').encode('utf-8'), line).group()
            #print(datetime.date(int(birth[:4]), int(birth[5:7]), int(birth[8:])))
            data.append((date - datetime.date(int(birth[:4]), int(birth[5:7]), int(birth[8:]))).days)
            participates = re.search(unicode(r'\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s', 'utf-8').encode('utf-8'), line).group().replace(',', '').split()
            dist_rec = get_distance_record_url(name, rcno, date)
            #print(participates)
            if int(participates[0]) == 0:
                data.extend([0, 0, 0, 0, 0])
            else:
                data.append(int(participates[0]))
                data.append(int(participates[1]))
                data.append(int(participates[2]))
                data.append(int(participates[1])*100/int(participates[0]))
                data.append(int(participates[2])*100/int(participates[0]))

            if int(participates[3]) == 0:
                data.extend([0, 0, 0, 0, 0])
            else:
                data.append(int(participates[3]))
                data.append(int(participates[4]))
                data.append(int(participates[5]))
                data.append(int(participates[4])*100/int(participates[3]))
                data.append(int(participates[5])*100/int(participates[3]))

            data.extend(dist_rec)
            assert len(data) == 17
            return data
    print("can not find %s in %s" % (name, filename))
    return [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]



# 이름  소속 생일        데뷔일  총경기수, 총1, 총2, 1년, 1년1, 1년2
# 김동철491974/11/28371995/07/015252 3706  217  242  166   17   19
def parse_txt_jockey(date, name):
    if len(str(name)) > 9:
        #print("name is changed %s -> %s" % (name, name[:6]))
        name = str(name)[:6]
    filename = get_fname(date, "jockey")
    if DEBUG: print(filename)
    f_input = open(filename)
    while True:
        line = f_input.readline()
        line = unicode(line, 'euc-kr').encode('utf-8')
        if len(line) == 0:
            break
        if re.search(name.encode('utf-8'), line) is not None:
            data = []
            participates = re.search(unicode(r'(?<=/\d{2})[ ,\d]+[\s\d]+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s', 'utf-8').encode('utf-8'), line).group().replace(',', '').split()
            #print(participates)
            if int(participates[0]) == 0:
                data.extend([0, 0, 0, 0, 0])
            else:
                data.append(int(participates[0]))
                data.append(int(participates[1]))
                data.append(int(participates[2]))
                data.append(int(participates[1])*100/int(participates[0]))
                data.append(int(participates[2])*100/int(participates[0]))

            if int(participates[3]) == 0:
                data.extend([0, 0, 0, 0, 0])
            else:
                data.append(int(participates[3]))
                data.append(int(participates[4]))
                data.append(int(participates[5]))
                data.append(int(participates[4])*100/int(participates[3]))
                data.append(int(participates[5])*100/int(participates[3]))

            return data
    print("can not find %s in %s" % (name, filename))
    return [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]


# 이름  소속 생일        데뷔일  총경기수, 총1, 총2, 1년, 1년1, 1년2
# 곽영효191961/09/24551997/05/283,868  438  394  134   18   13
def parse_txt_trainer(date, name):
    if len(str(name)) > 9:
        #print("name is changed %s -> %s" % (name, name[:6]))
        name = str(name)[:6]
    filename = get_fname(date, "trainer")
    if DEBUG: print(filename)
    f_input = open(filename)
    while True:
        line = f_input.readline()
        line = unicode(line, 'euc-kr').encode('utf-8')
        if len(line) == 0:
            break
        if re.search(name.encode('utf-8'), line) is not None:
            data = []
            participates = re.search(unicode(r'(?<=/\d\d)[\d,]+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+', 'utf-8').encode('utf-8'),
                                     line).group().replace(',', '').split()
            #print(participates)
            if int(participates[0]) == 0:
                data.extend([0, 0, 0, 0, 0])
            else:
                data.append(int(participates[0]))
                data.append(int(participates[1]))
                data.append(int(participates[2]))
                data.append(int(participates[1])*100/int(participates[0]))
                data.append(int(participates[2])*100/int(participates[0]))

            if int(participates[3]) == 0:
                data.extend([0, 0, 0, 0, 0])
            else:
                data.append(int(participates[3]))
                data.append(int(participates[4]))
                data.append(int(participates[5]))
                data.append(int(participates[4])*100/int(participates[3]))
                data.append(int(participates[5])*100/int(participates[3]))

            return data
    print("can not find %s in %s" % (name, filename))
    return [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]



def parse_xml_entry(meet, date, number):
    # get other data
    data_hr = xh.parse_xml_hr(meet)
    data_jk = xj.parse_xml_jk(meet)
    data_tr = xt.parse_xml_tr(meet)
    date_m = date / 100
    data = []
    filename = '../xml/entry/get_entry_%d_%d.xml' % (meet, date_m)
    file_input = open(filename)
    print "process in %s" % filename
    response_body = file_input.read()
    xml_text = BeautifulSoup(response_body, 'html.parser')
    humidity = get_humidity()
    for itemElm in xml_text.findAll('item'):
        #print itemElm
        rcdate = int("%s%s%s" % (itemElm.rcdate.string[:4], itemElm.rcdate.string[5:7], itemElm.rcdate.string[8:10]))
        rcno = int("%s" % (itemElm.rcno.string))
        if date != rcdate:
            continue
        if number > 0 and number != rcno:
            continue
        hr_gender, hr_days = get_hr_data(data_hr, itemElm.hrname.string)
        hr_weight, hr_dweight = get_hr_weight(meet, itemElm.rcdate.string, itemElm.rcno.string, itemElm.hrname.string)
        #hr_dist_rec = get_distance_record_url(itemElm.hrname.string, int(itemElm.rcno.string), date)
        cnt, kind = get_game_info(datetime.date(date/10000, date/100%100, date%100), int(itemElm.rcno.string))
        #hr_win = get_hr_win(itemElm.cntt.string, itemElm.ord1t.string, itemElm.ord2t.string, itemElm.cnty.string,
        #                   itemElm.ord1y.string, itemElm.ord2y.string)
        #jk_win = get_jk_win(data_jk, itemElm.jkname.string)
        #tr_win = get_tr_win(data_tr, itemElm.trname.string)
        hr_info = parse_txt_horse(datetime.date(date/10000, date/100%100, date%100), int(itemElm.rcno.string), itemElm.hrname.string)
        jk_win = parse_txt_jockey(datetime.date(date/10000, date/100%100, date%100), itemElm.jkname.string)
        tr_win = parse_txt_trainer(datetime.date(date/10000, date/100%100, date%100), itemElm.trname.string)

        hrname = itemElm.hrname.string
        dbudam = gdd.get_dbudam(2, date, int(rcno), hrname)
        drweight = gdd.get_drweight(2, date, int(rcno), hrname)
        lastday = gdd.get_lastday(2, date, int(rcno), hrname)
        train_state = gdd.get_train_state(2, date, int(rcno), hrname)

        adata = [itemElm.rcdist.string,
                 humidity,
                 kind,

                 dbudam,
                 drweight,
                 lastday,
                 train_state[0],
                 train_state[1],
                 train_state[2],
                 train_state[3],
                 train_state[4],

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
                 cnt,
                 itemElm.rcno.string,
                 date/100%100,
                 hr_info[0], #hr_days,

                 hr_info[1],  #itemElm.cntt.string,
                 hr_info[2],  #itemElm.ord1t.string,
                 hr_info[3],  #itemElm.ord2t.string,
                 hr_info[4],  #hr_win[0],
                 hr_info[5],  #hr_win[1],
                 hr_info[6],  #itemElm.cnty.string,
                 hr_info[7],  #itemElm.ord1y.string,
                 hr_info[8],  #itemElm.ord2y.string,
                 hr_info[9],  #hr_win[2],
                 hr_info[10],  #hr_win[3],

                 hr_info[11],  #hr_dist_rec[0],
                 hr_info[12],  #hr_dist_rec[1],
                 hr_info[13],  #hr_dist_rec[2],
                 hr_info[14],  #hr_dist_rec[3],
                 hr_info[15],  #hr_dist_rec[4],
                 hr_info[16],  #hr_dist_rec[5],

                 jk_win[0],
                 jk_win[1],
                 jk_win[2],
                 jk_win[3],
                 jk_win[4],
                 jk_win[5],
                 jk_win[6],
                 jk_win[7],
                 jk_win[8],
                 jk_win[9],

                 tr_win[0],
                 tr_win[1],
                 tr_win[2],
                 tr_win[3],
                 tr_win[4],
                 tr_win[5],
                 tr_win[6],
                 tr_win[7],
                 tr_win[8],
                 tr_win[9],
                ]
        #print(adata)
        data.append(adata)

    df = pd.DataFrame(data)
    df.columns = ['course', 'humidity', 'kind', 'dbudam', 'drweight', 'lastday', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'idx', 'name', 'cntry', 'gender', 'age', 'budam', 'jockey', 'trainer', 'owner', # 20
                  'weight', 'dweight', 'cnt', 'rcno', 'month', 'hr_days', 'hr_nt', 'hr_nt1', 'hr_nt2', 'hr_t1', 'hr_t2', 'hr_ny', 'hr_ny1',
                  'hr_ny2', 'hr_y1', 'hr_y2', 'hr_dt', 'hr_d1', 'hr_d2', 'hr_rh', 'hr_rm', 'hr_rl',
                  'jk_nt', 'jk_nt1', 'jk_nt2', 'jk_t1', 'jk_t2', 'jk_ny', 'jk_ny1',
                  'jk_ny2', 'jk_y1', 'jk_y2', 'tr_nt', 'tr_nt1', 'tr_nt2', 'tr_t1', 'tr_t2', 'tr_ny', 'tr_ny1',
                  'tr_ny2', 'tr_y1', 'tr_y2']
    return df


if __name__ == '__main__':
    DEBUG = True
    meet = 2
    rcno = 8
    date = 20161112
    data = parse_xml_entry(meet, date, rcno)
    print data

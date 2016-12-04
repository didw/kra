# -*- coding:utf-8 -*-

from urllib2 import urlopen
import os
from bs4 import BeautifulSoup
import datetime
import re
import numpy as np
from mean_data import mean_data

DEBUG = False


def get_budam(meet, date, rcno, name):
    name = name.replace('★', '')
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
            if name in itemList[1].string.encode('utf-8'):
                return unicode(itemList[6].string)
    print("can not find budam of %s in %s" % (name, fname))
    return -1


def get_dbudam(meet, date, rcno, name):
    name = name.replace('★', '')
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
            if len(itemList) >= 10 and name in itemList[1].string.encode('utf-8'):
                return unicode(itemList[7].string)
    print("can not find dbudam of %s in %s" % (name, fname))
    return 0


def get_weight(meet, date, rcno, name):
    name = name.replace('★', '')
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
            if name in itemList[1].string.encode('utf-8'):
                try:
                    return float(unicode(itemList[2].string))
                except ValueError:
                    return 465
    return 465


def get_dweight(meet, date, rcno, name):
    name = name.replace('★', '')
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
            if name in itemList[1].string.encode('utf-8'):
                return unicode(itemList[3].string)
    print("can not find dweight %s in %s" % (name, fname))
    return 0


def get_drweight(meet, date, rcno, name):
    name = name.replace('★', '')
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
            if len(itemList) < 5:
                continue
            if name in itemList[1].string.encode('utf-8'):
                last_date = itemList[4].string
                if len(last_date) >= 10:
                    last_date = datetime.date(int(last_date[:4]), int(last_date[5:7]), int(last_date[8:10]))
                    delta_day = datetime.date(date/10000, date/100%100, date%100) - last_date
                    return int(unicode(itemList[3].string)) * 1000 / delta_day.days
                else:
                    if "-R" not in last_date:
                        print("can not parsing get_drweight %s" % fname)
                    return 0
    print("can not find drweight %s in %s" % (name, fname))
    return 0


def get_lastday(meet, date, rcno, name):
    name = name.replace('★', '')
    fname = '../txt/%d/weight/weight_%d_%d_%d.txt' % (meet, meet, date, rcno)
    if DEBUG: print(fname)
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
            if len(itemList) < 5:
                continue
            if name in itemList[1].string.encode('utf-8'):
                last_date = itemList[4].string
                if len(last_date) >= 10:
                    last_date = datetime.date(int(last_date[:4]), int(last_date[5:7]), int(last_date[8:10]))
                    delta_day = datetime.date(date/10000, date/100%100, date%100) - last_date
                    return delta_day.days
                else:
                    if "-R" not in last_date:
                        print("can not parsing get_lastday %s" % fname)
                        return 29
                    else:
                        return 1000  # first attending
    print("can not find last day %s in %s" % (name, fname))
    return 29


def get_train_state(meet, date, rcno, name):
    name = name.replace('★', '')
    fname = '../txt/%d/train_state/train_state_%d_%d_%d.txt' % (meet, meet, date, rcno)
    res = [0, 0, 0, 0, 0, 0]
    cand = "조보후승기"
    if os.path.exists(fname):
        response_body = open(fname).read()
    else:
        base_url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfoTrainState.do?Act=02&Sub=1&"
        url = base_url + "meet=%d&rcNo=%d&rcDate=%d" % (meet, rcno, date)
        response_body = urlopen(url).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr'):
            itemList = itemElm2.findAll('td')
            if name in itemList[1].string.encode('utf-8'):
                for item in itemList[2:]:
                    if item.string is None:
                        continue
                    trainer = re.search(r'[가-힣]+', item.string.encode('utf-8'))
                    time = re.search(r'\d+', item.string.encode('utf-8'))
                    if trainer is None or time is None:
                        continue
                    trainer = trainer.group()
                    who = cand.find(trainer) / 3
                    if who == -1:
                        who = 4
                    train_time = int(re.search(r'\d+', item.string.encode('utf-8')).group())
                    res[who] += train_time
                    res[5] += train_time
                return res
    print("can not find train state %s in %s" % (name, fname))
    return [0, 0, 0, 0, 138, 155]


# http://race.kra.co.kr/racehorse/profileTrainState.do?Act=02&Sub=1&meet=1&hrNo=036114
def get_train_info(hridx):
    base_url = "http://race.kra.co.kr/racehorse/profileTrainState.do?Act=02&Sub=1&meet=1&hrNo="
    url = "%s%s" % (base_url, hridx)
    print(url)
    response_body = urlopen(url).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr'):
            if len(itemElm2) != 15:
                continue
            itemList = itemElm2.findAll('td')
            print(itemList[1].string)
            print(itemList[5].string)
    return -1


def get_distance_record(meet, name, rcno, date, course, md=mean_data()):
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
                    try:
                        return [0, 0, 0] + md.dist_rec[course][3:]
                    except KeyError:
                        print("there is no course %d" % course)
                        return [0, 0, 0, 0, 0, 0]
                if DEBUG:
                    print("%s, %s, %s, %s, %s, %s" % (unicode(itemList[2].string), unicode(itemList[3].string), unicode(itemList[4].string), unicode(itemList[5].string), unicode(itemList[6].string), unicode(itemList[7].string)))
                try:
                    cnt = re.search(r'\d+', unicode(itemList[2].string)).group()
                    res.append(int(cnt))
                    res.append(float(unicode(itemList[3].string)))
                    res.append(float(unicode(itemList[4].string)))
                    t = unicode(itemList[5].string)
                    res.append(int(t.split(':')[0]) * 600 + int(t.split(':')[1].split('.')[0]) * 10 + int(t.split('.')[1][0]))
                    t = unicode(itemList[6].string)
                    res.append(int(t.split(':')[0]) * 600 + int(t.split(':')[1].split('.')[0]) * 10 + int(t.split('.')[1][0]))
                    t = unicode(itemList[7].string)
                    res.append(int(t.split(':')[0]) * 600 + int(t.split(':')[1].split('.')[0]) * 10 + int(t.split('.')[1][0]))
                except:
                    break
    if len(res) == 6:
        return res
    else:
        print("can not find %s in %s" % (name, fname))
        try:
            return md.dist_rec[course]
        except KeyError:
            print("there is no course %d" % course)
            return [0, 0, 0, 0, 0, 0]


def get_hrno(meet, date, rcno, name):
    name = name.replace('★', '')
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
            try:
                hrname = itemList[1].string.encode('utf-8')
                hrname = hrname.replace('★', '')
            except:
                continue
            if name == hrname:
                return int(re.search(r'\d{6}', unicode(itemList[1])).group())
    print("can not find %s in fname %s" % (name, fname))
    return -1



def norm_racescore(meet, course, humidity, value, md=mean_data()):
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
    try:
        return value / md.race_score[course][humidity-1] * md.race_score[course][20]
    except KeyError:
        return value


def get_hr_racescore(meet, hrno, _date, mode='File', md=mean_data()):
    result = [-1, -1, -1, -1, -1, -1] # 주, 1000, 1200, 1300, 1400, 1700
    default_res = [md.race_score[1000][20], md.race_score[1000][20], md.race_score[1200][20], md.race_score[1300][20], md.race_score[1400][20], md.race_score[1700][20]]
    default_res.append(np.mean(default_res))
    race_sum = [[], [], [], [], [], []]
    if hrno == -1:
        return default_res
    fname = '../txt/%d/racehorse/racehorse_%d_%06d.txt' % (meet, meet, hrno)
    #print("racehorse: %s" % fname)
    if os.path.exists(fname) and mode == 'File':
        response_body = open(fname).read()
    else:
        base_url = "http://race.kra.co.kr/racehorse/profileRaceScore.do?Act=02&Sub=1&"
        url = base_url + "meet=%d&hrNo=%06d" % (meet, hrno)
        response_body = urlopen(url).read()
        print("open url %d" % hrno)
    try:
        xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    except UnicodeDecodeError:
        print("decode error: %s", fname)
        return default_res
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr')[2:]:
            itemList = itemElm2.findAll('td')
            #print(itemList)
            try:
                date = re.search(r'\d{4}/\d{2}/\d{2}', unicode(itemList[1])).group()
            except:
                continue
            date = int("%s%s%s" % (date[:4], date[5:7], date[8:]))
            if date > _date:
                continue
            if datetime.date(date/10000, date/100%100, date%100) + datetime.timedelta(days=365*3) < datetime.date(_date/10000, _date/100%100, _date%100):
                continue
            racekind = unicode(itemList[3].string).strip().encode('utf-8')
            try:
                distance = int(unicode(itemList[4].string).strip().encode('utf-8'))
            except:
                distance = 1000
            try:
                record = unicode(itemList[10].string).strip().encode('utf-8')
            except:
                continue
            #print(unicode(itemList[12].string).strip())
            try:
                humidity = int(re.search(r'\d+', unicode(itemList[12].string)).group())
            except AttributeError:
                humidity = 7
            try:
                record = int(record[0])*600 + int(record[2:4])*10 + int(record[5])
            except:
                continue
            if record == 0:
                continue
            #print("주, 일, %s" % racekind)
            record = norm_racescore(1, distance, humidity, record, md)
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
            #print("%d, %s, %s, %d" % (date, racekind, distance, record))

    for i in range(len(race_sum)):
        if len(race_sum[i]) == 0:
            #result[i] = -1
            result[i] = default_res[i]
        else:
            result[i] = np.mean(race_sum[i])
    result.append(np.mean(result))
    return result


if __name__ == '__main__':
    DEBUG = True
    print(get_lastday(1, 20070114, 1, "다시한번"))


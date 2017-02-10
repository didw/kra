# -*- coding:utf-8 -*-

from urllib2 import urlopen
import os
from bs4 import BeautifulSoup
import datetime
import re
import numpy as np
from mean_data import mean_data
import time

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
            if name in unicode(itemList[1].string).encode('utf-8'):
                return unicode(itemList[6].string)
    print("can not find budam of %s in %s" % (name, fname))
    return 54


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
            if len(itemList) >= 10 and name in unicode(itemList[1].string).encode('utf-8'):
                value = int(re.search(r'\d+', unicode(itemList[7].string)).group())
                return value
    print("can not find dbudam of %s in %s" % (name, fname))
    return 0


def get_weight(meet, date, rcno, name, course):
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
            if name in unicode(itemList[1].string).encode('utf-8'):
                try:
                    return float(float(unicode(itemList[2].string)))
                except ValueError:
                    return {400: 266, 800: 268, 900: 286, 1000: 293, 1110: 300, 1200: 300, 1400: 301, 1610: 302, 1700: 304, 1800: 304}[course]
    return {400: 266, 800: 268, 900: 286, 1000: 293, 1110: 300, 1200: 300, 1400: 301, 1610: 302, 1700: 304, 1800: 304}[course]


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
            if name in unicode(itemList[1].string).encode('utf-8'):
                return float(unicode(itemList[3].string))
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
            if name in unicode(itemList[1].string).encode('utf-8'):
                last_date = itemList[4].string
                if len(last_date) >= 10:
                    last_date = datetime.date(int(last_date[:4]), int(last_date[5:7]), int(last_date[8:10]))
                    delta_day = datetime.date(date/10000, date/100%100, date%100) - last_date
                    return float(unicode(itemList[3].string)) * 1000 / delta_day.days
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
            if name in unicode(itemList[1].string).encode('utf-8'):
                last_date = itemList[4].string
                if len(last_date) >= 10:
                    last_date = datetime.date(int(last_date[:4]), int(last_date[5:7]), int(last_date[8:10]))
                    delta_day = datetime.date(date/10000, date/100%100, date%100) - last_date
                    return delta_day.days
                else:  # first attending
                    if "-R" not in last_date:
                        print("can not parsing get_lastday %s" % fname)
                        return 20
                    else:
                        return 100  # first attending
    print("can not find last day %s in %s" % (name, fname))
    return 20


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
    if DEBUG: print("name: %s, %s" % (name, fname))
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr'):
            itemList = itemElm2.findAll('td')
            if name in unicode(itemList[1].string).encode('utf-8'):
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
            if len(itemList) < 5:
                continue
            #print("name: %s, %s" % (name, itemList[1].string.encode('utf-8')))
            if name in unicode(itemList[1].string).encode('utf-8'):
                #print("find name: %s, %s" % (name, itemList[1].string.encode('utf-8')))
                if int(unicode(itemList[2].string)[0]) == 0:
                    try:
                        return [0, 0, 0] + map(lambda x: float(x), md.dist_rec[course][3:])
                    except KeyError:
                        print("there is no course %d" % course)
                        return map(lambda x: float(x), md.dist_rec[course])
                if DEBUG:
                    print("%s, %s, %s, %s, %s, %s" % (unicode(itemList[2].string), unicode(itemList[3].string), unicode(itemList[4].string), unicode(itemList[5].string), unicode(itemList[6].string), unicode(itemList[7].string)))
                try:
                    cnt = re.search(r'\d+', unicode(itemList[2].string)).group()
                    res.append(float(cnt))
                    res.append(float(unicode(itemList[3].string)))
                    res.append(float(unicode(itemList[4].string)))
                    t = unicode(itemList[5].string)
                    res.append(float(t.split(':')[0]) * 600 + float(t.split(':')[1].split('.')[0]) * 10 + float(t.split('.')[1][0]))
                    t = unicode(itemList[6].string)
                    res.append(float(t.split(':')[0]) * 600 + float(t.split(':')[1].split('.')[0]) * 10 + float(t.split('.')[1][0]))
                    t = unicode(itemList[7].string)
                    res.append(float(t.split(':')[0]) * 600 + float(t.split(':')[1].split('.')[0]) * 10 + float(t.split('.')[1][0]))
                    break
                except:
                    print("parsing error")
                    break
    if len(res) == 6:
        return res
    else:
        print("can not find %s in %s" % (unicode(name, 'utf-8'), fname))
        try:
            return map(lambda x: float(x), md.dist_rec[course])
        except KeyError:
            print("there is no course %d" % course)
            return [-1, -1, -1, -1, -1, -1]


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
                hrname = unicode(itemList[1].string).encode('utf-8')
                hrname = hrname.replace('★', '')
            except:
                continue
            if name == hrname:
                return int(re.search(r'\d{6}', unicode(itemList[1])).group())
    print("can not find %s in fname %s" % (name, fname))
    return -1


def norm_racescore(course, month, humidity, value, md=mean_data()):
    humidity = min(humidity, 20) - 1
    try:
        return value * np.array(md.race_score[0])[:,20].mean() / md.race_score[0][month][humidity]
    except KeyError:
        return value


def get_hr_racescore(meet, hrno, _date, month, course, mode='File', md=mean_data()):
    first_attend = True
    course = int(course)
    result = [-1, -1, -1, -1, -1, -1, -1] # 주, 400, 800, 900, 1000, 1200, 0
    default_res = map(lambda x: float(np.mean(np.array(x)[:,20])), [md.race_score[300], md.race_score[400], md.race_score[800], md.race_score[900], md.race_score[1000], md.race_score[1200], md.race_score[0]])
    default_res.extend(map(lambda x: float(x), md.dist_rec[course][3:]))
    race_sum = [[], [], [], [], [], [], []]
    race_same_dist = []
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
        fout = open(fname, 'w')
        fout.write(response_body)
        fout.close()
        if os.path.getsize(fname) < 31100:
            os.remove(fname)
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
            if len(itemList) < 5:
                continue
            try:
                date = re.search(r'\d{4}/\d{2}/\d{2}', unicode(itemList[1])).group()
            except:
                print("regular expression error: %s" % itemList)
                continue
            date = int("%s%s%s" % (date[:4], date[5:7], date[8:]))
            month_ = date/100%100
            if date >= _date:
                continue

            if datetime.date(date/10000, date/100%100, date%100) + datetime.timedelta(days=365*1) < datetime.date(_date/10000, _date/100%100, _date%100):
                continue
            racekind = unicode(itemList[3].string).strip().encode('utf-8')
            try:
                distance = int(unicode(itemList[4].string).strip().encode('utf-8'))
            except:
                print("can not parsing distance")
                continue
            try:
                record = unicode(itemList[9].string).strip().encode('utf-8')
            except:
                print("unicode error")
                continue
            #print(unicode(itemList[9].string).strip())
            try:
                humidity = int(re.search(r'\d+', unicode(itemList[11].string)).group())
            except AttributeError:
                humidity = 7
            try:
                record = float(record[0])*600 + float(record[2:4])*10 + float(record[5])
            except:
                continue
            if record == 0:
                continue
            #print("주, 일, %s" % racekind)
            if racekind == '주':
                if distance == 1000:
                    distance = 300
                    record = record * md.race_score[800][month_-1][20] / md.race_score[1000][month_-1][20]
                if distance == 900:
                    distance = 300
                    record = float(record) * md.race_score[800][month_-1][20] / md.race_score[900][month_-1][20]
                if distance == 800:
                    distance = 300
            record = norm_racescore(distance, month_-1, humidity, record, md)
            if distance not in [300, 400, 800, 900, 1000, 1200]:
                continue
            if record < md.race_score[distance][month_-1][20]*0.8 or record > md.race_score[distance][month_-1][20]*1.2:
                continue
            if racekind == '주' and len(race_sum[0]) == 0:
                race_sum[0].append(record)
                race_sum[6].append(record * md.race_score[0][month_-1][humidity] / md.race_score[300][month_-1][humidity])
                if first_attend:
                    md.update_race_score_qual(month_-1, humidity, record)
                first_attend = False
            elif racekind == '일':
                if distance == 400:
                    race_sum[1].append(record)
                    race_sum[6].append(record * md.race_score[0][month_-1][humidity] / md.race_score[400][month_-1][humidity])
                elif distance == 800:
                    race_sum[2].append(record)
                    race_sum[6].append(record * md.race_score[0][month_-1][humidity] / md.race_score[800][month_-1][humidity])
                elif distance == 900:
                    race_sum[3].append(record)
                    race_sum[6].append(record * md.race_score[0][month_-1][humidity] / md.race_score[900][month_-1][humidity])
                elif distance == 1000:
                    race_sum[4].append(record)
                    race_sum[6].append(record * md.race_score[0][month_-1][humidity] / md.race_score[1000][month_-1][humidity])
                elif distance == 1200:
                    race_sum[5].append(record)
                    race_sum[6].append(record * md.race_score[0][month_-1][humidity] / md.race_score[1200][month_-1][humidity])
                if course == distance:
                    race_same_dist.append(record)
            #print("%d, %s, %s, %d" % (date, racekind, distance, record))


    course_score = [0,0,0,0,0,0,0]
    course_score[0] = np.mean(md.race_score[300])
    course_score[1] = np.mean(md.race_score[400])
    course_score[2] = np.mean(md.race_score[800])
    course_score[3] = np.mean(md.race_score[900])
    course_score[4] = np.mean(md.race_score[1000])
    course_score[5] = np.mean(md.race_score[1200])
    course_score[6] = np.mean(md.race_score[0])

    if len(race_sum[6]) == 0:
        result[6] = float(course_score[6])
    else:
        result[6] = np.mean(race_sum[6])
        race_sum[6].reverse()
        for r in race_sum[6]:
            result[6] += 0.5 * (r - result[6])  # v1, v2: 0.5, v3: 0.2
    for i in range(len(race_sum)-1):
        if len(race_sum[i]) == 0:
            #result[i] = -1
            result[i] = float(result[6] * course_score[i] / course_score[6])
        else:
            result[i] = np.mean(race_sum[i])
            race_sum[i].reverse()
            for r in race_sum[i]:
                result[i] += 0.5 * (r - result[i])  # v1, v2: 0.5, v3: 0.2
            result[i] = float(result[i])
    if len(race_same_dist) > 0:
        result.append(float(np.min(race_same_dist)))
        result.append(float(np.mean(race_same_dist)))
        result.append(float(np.max(race_same_dist)))
    elif course in [400, 800, 900, 1000, 1200]:
        #result.extend([-1, -1, -1])
        delta1 = md.dist_rec[course][4] - md.dist_rec[course][3]
        delta2 = md.dist_rec[course][5] - md.dist_rec[course][4]
        result.append(float(md.race_score[course][month-1][20] - delta1))
        result.append(float(md.race_score[course][month-1][20]))
        result.append(float(md.race_score[course][month-1][20] + delta2))
    else:
        #result.extend([-1, -1, -1])
        result.append(float(md.dist_rec[course][3]))
        result.append(float(md.dist_rec[course][4]))
        result.append(float(md.dist_rec[course][5]))
    return result


if __name__ == '__main__':
    DEBUG = True
    print(get_train_state(2, 20071006, 1, "삼다보배"))


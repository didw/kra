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
            if len(itemList) >= 10 and name in itemList[1].string.encode('utf-8'):
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
            if name in itemList[1].string.encode('utf-8'):
                try:
                    return int(float(unicode(itemList[2].string)))
                except ValueError:
                    return {}[course]
    return {}[course]


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
                        return 43
                    else:
                        return 100  # first attending
    print("can not find last day %s in %s" % (name, fname))
    return 43


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
    return [-1, -1, -1, -1, -1, -1]


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
            #print("name: %s, %s" % (name, itemList[1].string.encode('utf-8')))
            if name in itemList[1].string.encode('utf-8'):
                #print("find name: %s, %s" % (name, itemList[1].string.encode('utf-8')))
                if int(unicode(itemList[2].string)[0]) == 0:
                    try:
                        return [0, 0, 0] + map(lambda x: int(x), md.dist_rec[course][3:])
                    except KeyError:
                        print("there is no course %d" % course)
                        return map(lambda x: int(x), md.dist_rec[course])
                if DEBUG:
                    print("%s, %s, %s, %s, %s, %s" % (unicode(itemList[2].string), unicode(itemList[3].string), unicode(itemList[4].string), unicode(itemList[5].string), unicode(itemList[6].string), unicode(itemList[7].string)))
                try:
                    cnt = re.search(r'\d+', unicode(itemList[2].string)).group()
                    res.append(int(cnt))
                    res.append(int(float(unicode(itemList[3].string))))
                    res.append(int(float(unicode(itemList[4].string))))
                    t = unicode(itemList[5].string)
                    res.append(int(t.split(':')[0]) * 600 + int(t.split(':')[1].split('.')[0]) * 10 + int(t.split('.')[1][0]))
                    t = unicode(itemList[6].string)
                    res.append(int(t.split(':')[0]) * 600 + int(t.split(':')[1].split('.')[0]) * 10 + int(t.split('.')[1][0]))
                    t = unicode(itemList[7].string)
                    res.append(int(t.split(':')[0]) * 600 + int(t.split(':')[1].split('.')[0]) * 10 + int(t.split('.')[1][0]))
                    break
                except:
                    print("parsing error")
                    break
    if len(res) == 6:
        return res
    else:
        print("can not find %s in %s" % (name, fname))
        try:
            return map(lambda x: int(x), md.dist_rec[course])
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
                hrname = itemList[1].string.encode('utf-8')
                hrname = hrname.replace('★', '')
            except:
                continue
            if name == hrname:
                #print("hrname: %s, %d" % (name, int(re.search(r'\d{6}', unicode(itemList[1])).group())))
                return int(re.search(r'\d{6}', unicode(itemList[1])).group())
    print("can not find %s in fname %s" % (name, fname))
    return -1



def norm_racescore(meet, course, humidity, value, md=mean_data()):
    humidity = min(humidity, 20) - 1
    try:
        return value * md.race_score[0][20] / md.race_score[0][humidity]
    except KeyError:
        return value


def get_hr_racescore(meet, hrno, _date, course, mode='File', md=mean_data()):
    first_attend = True
    course = int(course)
    result = [-1, -1, -1, -1, -1, -1, -1] # 주, 1000, 1200, 1300, 1400, 1600, 0
    default_res = map(lambda x: int(x), [md.race_score[900][20], md.race_score[1000][20], md.race_score[1200][20], md.race_score[1300][20], md.race_score[1400][20], md.race_score[1600][20], md.race_score[0][20]])
    default_res.extend(map(lambda x: int(x), md.dist_rec[course][3:]))
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
            try:
                date = re.search(r'\d{4}/\d{2}/\d{2}', unicode(itemList[1])).group()
            except:
                print("regular expression error")
                continue
            date = int("%s%s%s" % (date[:4], date[5:7], date[8:]))
            if date >= _date:
                continue

            if datetime.date(date/10000, date/100%100, date%100) + datetime.timedelta(days=365*1) < datetime.date(_date/10000, _date/100%100, _date%100):
                continue
            racekind = unicode(itemList[3].string).strip().encode('utf-8')
            try:
                distance = int(unicode(itemList[4].string).strip().encode('utf-8'))
            except:
                if racekind == '주':
                    distance = 900
                else:
                    continue
            try:
                record = unicode(itemList[10].string).strip().encode('utf-8')
            except:
                print("unicode error")
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
            if distance not in []:
                continue
            if record < md.race_score[distance][20]*0.8 or record > md.race_score[distance][20]*1.2:
                continue
            if racekind == '주' and len(race_sum[0]) == 0:
                race_sum[0].append(record)
                race_sum[6].append(record * md.course_record[6] / md.course_record[0])
                if first_attend:
                    md.update_race_score_qual(humidity, record)
                first_attend = False
            elif racekind == '일':
                if distance == 1000:
                    race_sum[1].append(record)
                    race_sum[6].append(record * md.course_record[6] / md.course_record[1])
                elif distance == 1200:
                    race_sum[2].append(record)
                    race_sum[6].append(record * md.course_record[6] / md.course_record[2])
                elif distance == 1300:
                    race_sum[3].append(record)
                    race_sum[6].append(record * md.course_record[6] / md.course_record[3])
                elif distance == 1400:
                    race_sum[4].append(record)
                    race_sum[6].append(record * md.course_record[6] / md.course_record[4])
                elif distance == 1700:
                    race_sum[5].append(record)
                    race_sum[6].append(record * md.course_record[6] / md.course_record[5])
                if course == distance:
                    race_same_dist.append(record)
            #print("%d, %s, %s, %d" % (date, racekind, distance, record))


    if len(race_sum[6]) != 0:
        result[6] = int(np.mean(race_sum[6]))
    else:
        result[6] = int(md.course_record[6])
    for i in range(len(race_sum)):
        if len(race_sum[i]) == 0:
            result[i] = int(result[6] * md.course_record[i] / md.course_record[6])
        else:
            result[i] = int(np.mean(race_sum[i]))
    if len(race_same_dist) > 0:
        result.append(int(np.min(race_same_dist)))
        result.append(int(np.mean(race_same_dist)))
        result.append(int(np.max(race_same_dist)))
    elif course in [1000, 1200, 1300, 1400, 1700]:
        #result.extend([-1, -1, -1])
        delta1 = md.dist_rec[course][4] - md.dist_rec[course][3]
        delta2 = md.dist_rec[course][5] - md.dist_rec[course][4]
        result.append(int(md.race_score[course][20] - delta1))
        result.append(int(md.race_score[course][20]))
        result.append(int(md.race_score[course][20] + delta2))
    else:
        #result.extend([-1, -1, -1])
        result.append(int(md.dist_rec[course][3]))
        result.append(int(md.dist_rec[course][4]))
        result.append(int(md.dist_rec[course][5]))
    return result


if __name__ == '__main__':
    DEBUG = True
    print(get_dbudam(3, 20080718, 5, "강공드라이브"))


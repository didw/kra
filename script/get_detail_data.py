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
            if len(itemList) == 14:
                ret_idx = 6
            elif len(itemList) == 15:
                ret_idx = 7
            else:
                continue
            if name in unicode(itemList[1].string).encode('utf-8'):
                return unicode(itemList[ret_idx].string)
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
            if len(itemList) == 14:
                ret_idx = 7
            elif len(itemList) == 15:
                ret_idx = 8
            else:
                continue
            if len(itemList) >= 10 and name in unicode(itemList[1].string).encode('utf-8'):
                value = int(re.search(r'[-\d]+', unicode(itemList[ret_idx].string)).group())
                return value
    print("can not find dbudam of %s in %s" % (name, fname))
    return 0


def get_weight(meet, date, rcno, name, course):
    name = name.replace('★', '')
    fname = '../txt/%d/weight/weight_%d_%d_%d.txt' % (meet, meet, date, rcno)
    if os.path.exists(fname) and True:
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
                    return float(unicode(itemList[2].string))
                except ValueError:
                    print("could not convert string to float %s, %s" % (name, unicode(itemList[2].string)))
                    continue
    return -1
    return {400: 266, 800: 268, 900: 286, 1000: 293, 1110: 300, 1200: 300, 1400: 301, 1610: 302, 1700: 304, 1800: 304}[course]


def get_dweight(meet, date, rcno, name):
    name = name.replace('★', '')
    fname = '../txt/%d/weight/weight_%d_%d_%d.txt' % (meet, meet, date, rcno)
    if os.path.exists(fname) and True:
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
                    return float(unicode(itemList[3].string))
                except ValueError:
                    print("could not convert string to float %s, %s" % (name, unicode(itemList[2].string)))
                    continue
    print("can not find dweight %s in %s" % (name, fname))
    return 0


def get_drweight(meet, date, rcno, name):
    name = name.replace('★', '')
    fname = '../txt/%d/weight/weight_%d_%d_%d.txt' % (meet, meet, date, rcno)
    if os.path.exists(fname) and True:
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
        try:
            response_body = urlopen(url).read()
        except httplib.ImcompleteRead, e:
            response_body = e.partial
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for itemElm in xml_text.findAll('tbody'):
        for itemElm2 in itemElm.findAll('tr'):
            itemList = itemElm2.findAll('td')
            if len(itemList) < 5:
                continue
            if name in unicode(itemList[1].string).encode('utf-8'):
                last_date = itemList[4].string
                if last_date is None:
                    return 20
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
    res = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
                try:
                    int(unicode(itemList[2].string)[0])
                except ValueError:
                    print("ValueError: ", unicode(itemList[2].string))
                    return [0, 0, 0] + map(float, md.dist_rec[course][3:])
                if int(unicode(itemList[2].string)[0]) == 0:
                    try:
                        return [0, 0, 0] + map(float, md.dist_rec[course][3:])
                    except KeyError:
                        print("there is no course %d" % course)
                        return map(float, md.dist_rec[course])
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
            return map(float, md.dist_rec[course])
        except KeyError:
            print("there is no course %d" % course)
            return [0, 0, 0] + map(float, md.dist_rec[course][3:])


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


def norm_racescore(value, md, hr_data, name, course):
    key_idx = {'humidity':0, 'month':22, 'age':9, 'idx':6, 'rcno':21, 'budam':10, 'dbudam':2, 'jockey':11, 'trainer':12}
    for i in range(1,82):
        key_idx['jc%d'%i] = 23+i
    printed = False
    for column in key_idx:
        try:
            if md[column][hr_data[key_idx[column]]] > 1.1 or md[column][hr_data[key_idx[column]]] < 0.9:
                print("mean data should be in between 0.9 and 1.1")
            value /= md[column][hr_data[key_idx[column]]]
        except KeyError:
            if not printed: print("name: %s, course: %d, column %s, hr_data: %s is not exists" % (name, course, str(column), hr_data[key_idx[column]]))
            printed = True
            pass
    try:
        value /= md['hr_days'][int(hr_data[5]/100)*100]
    except KeyError:
        if not printed: print("name: %s, course: %d, column %s, hr_data: %d is not exists" % (name, course, "hr_days", int(hr_data[5]/100)*100))
        printed = True
        pass
    except TypeError:
        print("name: %s, course: %d, column %s, hr_data[hr_days] is NoneType" % (name, course, "hr_days",))
        pass
    try:
        value /= md['lastday'][int(hr_data[4]/10)*10]
    except KeyError:
        if not printed: print("column %s, hr_data: %d is not exists" % ("lastday", int(hr_data[4]/10)*10))
        printed = True
        pass
    return value


def make_mean_race_record(race_record):
    res_dict = {}
    weight_list = []
    #400: 266, 800: 268, 900: 286, 1000: 293, 1110: 300, 1200: 300, 1400: 301, 1610: 302, 1700: 304, 1800: 304
    for course in [400, 800, 900, 1000, 1110, 1700]:
        res = [0]*4
        res_list = [[] for _ in range(4)]
        for name in race_record.data.keys():
            if course not in race_record.data[name]:
                continue
            for data in race_record.data[name][course]:
                res_list[0].append(data[17])
                res_list[1].append(data[18])
                res_list[2].append(data[19])
                res_list[3].append(data[16])
                weight_list.append(data[14])
        for i in range(4):
            res[i] = np.mean(res_list[i])
        res_dict[course] = res
    return res_dict, np.mean(weight_list)


def get_hr_race_record_mean(hrname, date, race_record, md):
    mean_data, weight = make_mean_race_record(race_record)
    res = []
    weight_list = []
    res_summary = [[] for _ in range(4)]
    for course in [300, 800, 900, 1000, 1200, 1400]:
        res.extend(mean_data[course])
        for i in range(4):
            res_summary[i].append(mean_data[course][i])
    for i in range(4):
        res.append(np.mean(res_summary[i]))  # default result

    try:
        hr_data = race_record.data[hrname]
    except KeyError:
        print("No hrname:%s, return default"%hrname)
        return res, weight
    res_summary = [[] for _ in range(4)]
    for ic, course in enumerate([300, 800, 900, 1000, 1200, 1400]):
        res_list = [[] for _ in range(4)]
        if course not in hr_data.keys():
            continue
        for hr_c_data in hr_data[course]:
            if hr_c_data[23] >= date:
                continue
            res_list[0].append(norm_racescore(hr_c_data[17], md, hr_c_data, hrname, course))
            res_list[1].append(norm_racescore(hr_c_data[18], md, hr_c_data, hrname, course))
            res_list[2].append(norm_racescore(hr_c_data[19], md, hr_c_data, hrname, course))
            res_list[3].append(norm_racescore(hr_c_data[16], md, hr_c_data, hrname, course))
            res_summary[0].append(norm_racescore(hr_c_data[17], md, hr_c_data, hrname, course)/mean_data[course][0])
            res_summary[1].append(norm_racescore(hr_c_data[18], md, hr_c_data, hrname, course)/mean_data[course][1])
            res_summary[2].append(norm_racescore(hr_c_data[19], md, hr_c_data, hrname, course)/mean_data[course][2])
            res_summary[3].append(norm_racescore(hr_c_data[16], md, hr_c_data, hrname, course)/mean_data[course][3])
            weight_list.append(hr_c_data[14])
        for i in range(4):
            if len(res_list[i]) == 0:
                continue
            res[ic*4+i] = np.mean(res_list[i])
    for i in range(4):
        if len(res_summary[i]) == 0:
            continue
        res[6*4+i] = np.mean(res_summary[i])
    return res, np.mean(weight_list)


def get_hr_race_record(hrname, date, race_record, md):
    mean_data, weight = make_mean_race_record(race_record)
    default_res = []
    weight_list = []
    res_summary = [[] for _ in range(4)]
    for course in [300, 800, 900, 1000, 1200, 1400]:
        default_res.extend(mean_data[course])
        for i in range(4):
            res_summary[i].append(mean_data[course][i])
    for i in range(4):
        default_res.append(np.mean(res_summary[i]))  # default result

    try:
        hr_data = race_record.data[hrname]
    except KeyError:
        print("No hrname:%s, return default"%hrname)
        return default_res, weight
    res_summary = [[] for _ in range(4)]
    for ic, course in enumerate([300, 800, 900, 1000, 1200, 1400]):
        if course not in hr_data.keys():
            continue
        for hr_c_data in hr_data[course]:
            if hr_c_data[23] >= date:
                continue
            res_summary[0].append(norm_racescore(hr_c_data[17], md, hr_c_data, hrname, course)/mean_data[course][0])
            res_summary[1].append(norm_racescore(hr_c_data[18], md, hr_c_data, hrname, course)/mean_data[course][1])
            res_summary[2].append(norm_racescore(hr_c_data[19], md, hr_c_data, hrname, course)/mean_data[course][2])
            res_summary[3].append(norm_racescore(hr_c_data[16], md, hr_c_data, hrname, course)/mean_data[course][3])

    res = [0]*28
    for i in range(4):
        if len(res_summary[i]) == 0:
            res[6*4+i] = default_res[6*4+i]
        else:
            res[6*4+i] = np.mean(res_summary[i]) * default_res[6*4+i]
    for ic, course in enumerate([300, 800, 900, 1000, 1200, 1400]):
        res_list = [[] for _ in range(4)]
        if course not in hr_data.keys():
            for i in range(4):
                res[ic*4+i] = res[6*4+i] * default_res[ic*4+i] / default_res[6*4+i]
            continue
        for hr_c_data in hr_data[course]:
            if hr_c_data[23] >= date:
                continue
            res_list[0].append(norm_racescore(hr_c_data[17], md, hr_c_data, hrname, course))
            res_list[1].append(norm_racescore(hr_c_data[18], md, hr_c_data, hrname, course))
            res_list[2].append(norm_racescore(hr_c_data[19], md, hr_c_data, hrname, course))
            res_list[3].append(norm_racescore(hr_c_data[16], md, hr_c_data, hrname, course))
            weight_list.append(hr_c_data[14])
        for i in range(4):
            if len(res_list[i]) == 0:
                res[ic*4+i] = res[6*4+i] * default_res[ic*4+i] / default_res[6*4+i]
                continue
            res[ic*4+i] = np.mean(res_list[i])
    for i in range(len(res)):
        if res[i] < 100:
            print("res[i]: %d is weired.." % res[i])
    return res, np.mean(weight_list)


if __name__ == '__main__':
    DEBUG = True
    print(get_train_state(2, 20071006, 1, "삼다보배"))


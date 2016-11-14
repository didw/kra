# -*- coding:utf-8 -*-

from urllib2 import urlopen
import os
from bs4 import BeautifulSoup
import datetime
import re


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
            print(itemElm2)
            itemList = itemElm2.findAll('td')
            if name == itemList[1].string.encode('utf-8'):
                return itemList[6].string
    return -1


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


def get_train_state(meet, date, rcno, name):
    fname = '../txt/%d/train_state/train_state_%d_%d_%d.txt' % (meet, meet, date, rcno)
    res = [-1, -1, -1, -1, -1]
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
            if name == itemList[1].string.encode('utf-8'):
                for item in itemList[2:]:
                    if item.string is None:
                        continue
                    trainer = re.search(r'[가-힣]+', item.string.encode('utf-8')).group()
                    who = cand.find(trainer) / 3
                    if who == -1:
                        who = 4
                    train_time = int(re.search(r'\d+', item.string.encode('utf-8')).group())
                    res[who] += train_time
    return res


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



if __name__ == '__main__':
    print(get_train_state(1, 20161113, 1, "파인챔프"))


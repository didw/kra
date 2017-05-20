# -*- coding:utf-8 -*-

from urllib2 import urlopen
import os
from bs4 import BeautifulSoup
import datetime
import re
import numpy as np
from mean_data import mean_data
import time
import glob

DEBUG = False



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
                #print("hrname: %s, %d" % (name, int(re.search(r'\d{6}', unicode(itemList[1])).group())))
                return int(re.search(r'\d{6}', unicode(itemList[1])).group())
    print("can not find %s in fname %s" % (name, fname))
    return -1


def get_lineage(hrno, lineage):
    meet = 1
    fname = "../txt/%d/LineageInfo/LineageInfo_%d_%06d.txt" % (meet, meet, hrno)
    print("processing %s" % fname)
    response_body = open(fname).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    idx = 0
    for i1, itemElm in enumerate(xml_text.findAll('tbody')[1:]):
        for i2, itemElm2 in enumerate(itemElm.findAll('tr')):
            itemList = itemElm2.findAll('td')
            for items in itemList:
                item = items.findAll('span')
                #print(item[0].string)
                try:
                    lineage[idx][unicode(item[0].string)] += 1
                except KeyError:
                    lineage[idx][unicode(item[0].string)] = 1
                idx += 1

def analyse_dict(data):
    for i in range(len(data)):
        print(i, len(data[i]))

if __name__ == '__main__':
    lineage = [dict() for _ in range(62)]
    #get_lineage(24004, lineage)
    flist = glob.glob('../txt/1/LineageInfo/*')
    for fname in flist[1000:2000]:
        get_lineage(int(fname[-10:-4]), lineage)
    analyse_dict(lineage)


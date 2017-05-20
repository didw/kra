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


lineage = [dict() for _ in range(32)]
def get_lineage(hrno):
    meet = 1
    fname = "../txt/%d/LineageInfo/LineageInfo_%d_%06d.txt" % (meet, meet, hrno)
    print("processing %s" % fname)
    response_body = open(fname).read()
    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    for i, itemElm in enumerate(xml_text.findAll('tbody')[1:]):
        for i, itemElm2 in enumerate(itemElm.findAll('tr')):
            itemList = itemElm2.findAll('td')
            for items in itemList:
                item = items.findAll('span')
                try:
                    lineage[i][unicode(item[0].string)] += 1
                except KeyError:
                    lineage[i][unicode(item[0].string)] = 1


if __name__ == '__main__':
    flist = glob.glob('../txt/1/LineageInfo/*')
    for fname in flist:
        get_lineage(int(fname[-10:-4]))
    print(lineage)


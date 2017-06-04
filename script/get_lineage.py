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
from sklearn.externals import joblib

DEBUG = False


def add_lineage_info(meet, hrno, lineage):
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
                if unicode(item[0].string) not in lineage[idx]:
                    lineage[idx].append(unicode(item[0].string))
                idx += 1


def save_lineage_info(meet):
    lineage = [[] for _ in range(62)]
    flist = glob.glob('../txt/%d/LineageInfo/*'%meet)
    for fname in flist:
        add_lineage_info(meet, int(fname[-10:-4]), lineage)
    joblib.dump(lineage, '../data/lineage_1.pkl')


def load_lineage_info(meet):
    lineage = joblib.load('../data/lineage_1.pkl')
    for i in range(len(lineage)):
        print(i, len(lineage[i]))


def get_lineage(meet, hrno, mode='File'):
    lineage = []*62
    md = joblib.load('../data/lineage_%d.pkl'%meet)
    fname = "../txt/%d/LineageInfo/LineageInfo_%d_%06d.txt" % (meet, meet, hrno)
    print("processing %s" % fname)
    if os.path.exists(fname) and mode == 'File':
        response_body = open(fname).read()
    else:
        base_url = "http://race.kra.co.kr/racehorse/profileLineageInfo.do?Act=02&Sub=1&"
        url = base_url + "meet=%d&hrNo=%06d" % (meet, hrno)
        response_body = urlopen(url).read()
        fout = open(fname, 'w')
        fout.write(response_body)
        fout.close()
        if os.path.getsize(fname) < 43000:
            os.remove(fname)
        print("open url %d" % hrno)

    xml_text = BeautifulSoup(response_body.decode('euc-kr'), 'html.parser')
    idx = 0
    for i1, itemElm in enumerate(xml_text.findAll('tbody')[1:]):
        for i2, itemElm2 in enumerate(itemElm.findAll('tr')):
            itemList = itemElm2.findAll('td')
            for items in itemList:
                item = items.findAll('span')
                lineage[idx] = md[idx].index(item[0].string)
                idx += 1
    return lineage


if __name__ == '__main__':
    save_lineage_info(1)
    load_lineage_info(1)


# -*- coding:utf-8 -*-
from __future__ import print_function
import requests
import os
from bs4 import BeautifulSoup
import datetime
import re
import numpy as np
from mean_data import mean_data
import time
import glob
import sys
from sklearn.externals import joblib
from etaprogress.progress import ProgressBar
import _pickle, gzip

DEBUG = False


def add_lineage_info(meet, hrno, lineage):
    fname = "../txt/%d/LineageInfo/LineageInfo_%d_%06d.txt" % (meet, meet, hrno)
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
    bar = ProgressBar(len(flist), max_width=80)
    for fname in flist:
        bar.numerator += 1
        add_lineage_info(meet, int(fname[-10:-4]), lineage)
        print("%s" % (bar,), end='\r')
        sys.stdout.flush()
    serialized = _pickle.dumps(lineage)
    with gzip.open('../data/lineage_1.gz', 'wb') as f:
        f.write(serialized)


def load_lineage_info(meet):
    with gzip.open('../data/lineage_1.gz') as f:
        lineage = _pickle.loads(f.read())
    for i in range(len(lineage)):
        print(i, len(lineage[i]))


def get_lineage(meet, hrno, mode='File'):
    lineage = [-1]*62
    if hrno == -1:
        print("there's no  matching lineage")
        return lineage
    #md = joblib.load('../data/lineage_%d.pkl'%meet)
    with gzip.open('../data/lineage_1.gz') as f:
        md = _pickle.loads(f.read())
    fname = "../txt/%d/LineageInfo/LineageInfo_%d_%06d.txt" % (meet, meet, hrno)
    #print("processing %s" % fname)
    if os.path.exists(fname) and mode == 'File':
        response_body = open(fname).read()
    else:
        if not os.path.exists("../txt/%d/LineageInfo/" % meet):
            os.makedirs("../txt/%d/LineageInfo/" % meet)
        base_url = "http://race.kra.co.kr/racehorse/profileLineageInfo.do?Act=02&Sub=1&"
        url = base_url + "meet=%d&hrNo=%06d" % (meet, hrno)
        response_body = requests.get(url).text
        fout = open(fname, 'w')
        fout.write(response_body)
        fout.close()
        if os.path.getsize(fname) < 43000:
            os.remove(fname)
        print("open url %d" % hrno)

    xml_text = BeautifulSoup(response_body, 'html.parser')
    idx = 0
    printed = False
    for i1, itemElm in enumerate(xml_text.findAll('tbody')[1:]):
        for i2, itemElm2 in enumerate(itemElm.findAll('tr')):
            itemList = itemElm2.findAll('td')
            for items in itemList:
                item = items.findAll('span')
                try:
                    lineage[idx] = md[idx].index(item[0].string)
                except ValueError:
                    if not printed:
                        print("ValueError in %s, Set to -1" % fname)
                        printed = True
                    lineage[idx] = -1
                idx += 1
    return lineage


if __name__ == '__main__':
    save_lineage_info(1)
    #load_lineage_info(1)


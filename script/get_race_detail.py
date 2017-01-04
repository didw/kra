# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os.path
import glob
from sklearn.externals import joblib

def printu(data):
    print(unicode(data, 'utf-8'))

DEBUG = False
class RaceDetail:
    def __init__(self):
        self.data = dict()
    def parse_race_detail(self, filename):
        in_data = open(filename)
        data = dict()
        while True:
            line = in_data.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if line is None or len(line) == 0:
                break
            # 제목 : 16년12월25일(일)  제15경주
            date = int(re.search(r'\d{8}', filename).group())
            if re.search(r'\(서울\)', line) is not None:
                course = int(re.search(r'\d+00*(?=M)', line).group())
                #print("course: %d" % course)
            # read name
            if re.search(r'부담중량', line) is not None and re.search(r'산지', line) is not None:
                name_list = []
                while re.search(r'[-─]{5}', line) is None:
                    line = in_data.readline()
                    break
                while True:
                    line = in_data.readline()
                    line = unicode(line, 'euc-kr').encode('utf-8')
                    if re.search(r'[-─]{5}', line) is not None:
                        res = re.search(r'[-─]{5}', line).group()
                        if DEBUG: print("break: %s" % res)
                        break
                    try:
                        name = re.findall(r'\S+', line)[2]
                    except:
                        print("except: in %s : %s" % (filename, line))
                    name_list.append(name)
                    data[name] = [date, course]
                    if DEBUG:
                        print("name: %s" % unicode(name, 'utf-8'))

            # read score
            if re.search(r'단승식', line) is not None:
                while re.search(r'[-─]{10}', line) is None:
                    line = in_data.readline()
                    break
                i = 0
                while True:
                    line = in_data.readline()
                    line = unicode(line, 'euc-kr').encode('utf-8')
                    if re.search(r'[-─]{10}', line) is not None:
                        res = re.search(r'[-─]{10}', line).group()
                        #print("result: %s" % res)
                        break
                    words = re.findall(r'\S+', line)
                    if len(words) == 9:
                        g1f = words[6]
                    elif len(words) == 11:
                        g1f = words[8]
                    elif len(words) < 9:
                        #print("unexpected line: %s" % line)
                        continue
                    s1f = words[3]
                    g3f = words[2]
                    if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (s1f, g1f, g3f))
                    data[name_list[i]].extend([s1f, g1f, g3f])
                    i += 1
                    
        for k,v in data.iteritems():
            if self.data.has_key(k):
                self.data[k].append(v)
            else:
                self.data[k] = [v]


if __name__ == '__main__':
    DEBUG = False
    rd = RaceDetail()
    for year in range(2007,2017):
        filelist = glob.glob('../txt/1/rcresult/rcresult_1_%d*.txt' % year)
        for fname in filelist:
            rd.parse_race_detail(fname)
            print("processed in %s" % fname)
    #fname = '../txt/1/rcresult/rcresult_1_20071007.txt'
    #rd.parse_race_detail(fname)
    #for k,v in rd.data.iteritems():
    #    print(unicode(k, 'utf-8'))
    #    print(v)
    joblib.dump(rd, '../data/1_2007_2016_rd.pkl')


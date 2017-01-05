# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os.path
import glob
from sklearn.externals import joblib
from mean_data import mean_data

def printu(data):
    print(unicode(data, 'utf-8'))

DEBUG = False
class RaceDetail:
    def __init__(self):
        # [date, course, s1f, g1f, g3f]
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
                        name = re.findall(r'\S+', line)[2].replace('★', '')
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
            if len(v) < 5:
                continue
            if self.data.has_key(k):
                self.data[k].append(v)
            else:
                self.data[k] = [v]

    def parse_ap_rslt(self, filename):
        in_data = open(filename)
        data = dict()
        course = 900
        while True:
            line = in_data.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if line is None or len(line) == 0:
                break
            # 제목 : 16년12월25일(일)  제15경주
            date = int(re.search(r'\d{8}', filename).group())
            # read name
            if re.search(r'산지', line) is not None and (re.search(r'선수명', line) is not None or re.search(r'기수명', line) is not None):
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
                        name = re.findall(r'\S+', line)[2].replace('★', '')
                    except:
                        print("except: in %s : %s" % (filename, line))
                    name_list.append(name)
                    data[name] = [date, course]
                    if DEBUG:
                        print("name: %s" % unicode(name, 'utf-8'))

            # read score
            if re.search(r'S-1F', line) is not None:
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
                    if len(words) < 9:
                        #print("unexpected line: %s" % line)
                        continue
                    s1f = words[3]
                    g1f = words[6]
                    g3f = words[2]
                    if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (s1f, g1f, g3f))
                    data[name_list[i]].extend([s1f, g1f, g3f])
                    i += 1
        for k,v in data.iteritems():
            if len(v) < 5:
                continue
            if k in self.data:
                self.data[k].append(v)
            else:
                self.data[k] = [v]
    def get_data(self, name, date, md=mean_data()):
        res = []
        rs = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        course_list = [900, 1000, 1200, 1300, 1400, 1700]
        for c in range(len(course_list)):
            for data in self.data[name]:
                if data[0] < date and data[1] == c:
                    rs[3*c+0].append(data[2])
                    rs[3*c+1].append(data[3])
                    rs[3*c+2].append(data[4])
        for i in range(len(rs)):
            if len(rs[i]) == 0:
                res.append(md.race_detail[course_list[i/3]][i%3])
            else:
                res.append(np.mean(rs[i]))
        for i in range(len(rs)):
            rs[i].reverse()
            for j in rs[i]:
                res[i] += 0.1*(rs[i][j] - res[i])
        return res  # len: 18


if __name__ == '__main__':
    DEBUG = False
    rd = RaceDetail()
    for year in range(2007,2017):
        filelist1 = glob.glob('../txt/1/ap-check-rslt/ap-check-rslt_1_%d*.txt' % year)
        filelist2 = glob.glob('../txt/1/rcresult/rcresult_1_%d*.txt' % year)
        for fname in filelist1:
            print("processed ap %s" % fname)
            rd.parse_ap_rslt(fname)
        for fname in filelist2:
            print("processed rc in %s" % fname)
            rd.parse_race_detail(fname)
    #fname1 = '../txt/1/ap-check-rslt/ap-check-rslt_1_20130308.txt'
    #fname2 = '../txt/1/rcresult/rcresult_1_20110611.txt'
    #rd.parse_ap_rslt(fname1)
    #rd.parse_race_detail(fname2)
    #for k,v in rd.data.iteritems():
    #    print(unicode(k, 'utf-8'))
    #    print(v)
    joblib.dump(rd, '../data/1_2007_2016_rd.pkl')


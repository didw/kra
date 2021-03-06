# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os.path
import glob
from sklearn.externals import joblib
from mean_data import mean_data
from operator import itemgetter

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
            #line = unicode(line, 'euc-kr').encode('utf-8')
            if line is None or len(line) == 0:
                break
            # 제목 : 16년12월25일(일)  제15경주
            date = int(re.search(r'\d{8}', filename).group())
            if re.search(r'\(서울\)', line) is not None:
                course = int(re.search(r'\d+00*(?=M)', line).group())
            if re.search(r'경주조건', line) is not None and re.search(r'주로', line) is not None:
                try:
                    humidity = int(re.search(r'(?<=\()[\d ]+(?=\%\))', line).group())
                except:
                    humidity = 10
            # read name
            if re.search(r'부담중량', line) is not None and re.search(r'산지', line) is not None:
                name_list = []
                while re.search(r'[-─]{5}', line) is None:
                    line = in_data.readline()
                    break
                while True:
                    line = in_data.readline()
                    #line = unicode(line, 'euc-kr').encode('utf-8')
                    if re.search(r'[-─]{5}', line) is not None:
                        res = re.search(r'[-─]{5}', line).group()
                        if DEBUG: print("break: %s" % res)
                        break
                    try:
                        name = re.findall(r'\S+', line)[2].replace('★', '')
                    except:
                        print("except: in %s : %s" % (filename, line))
                    name_list.append(name)
                    data[name] = [date, course, humidity]
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
                    #line = unicode(line, 'euc-kr').encode('utf-8')
                    if re.search(r'[-─]{10}', line) is not None:
                        res = re.search(r'[-─]{10}', line).group()
                        #print("result: %s" % res)
                        break
                    words = re.findall(r'\S+', line)
                    s1f, g1f, g2f, g3f = -1, -1, -1, -1
                    if len(words) == 9:
                        if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (words[3], words[6], words[2]))
                        try:
                            g1f = float(re.search(r'\d{2}\.\d', words[6]).group())*10
                        except:
                            print("parsing error in race_detail - 1")
                            g1f = -1
                    elif len(words) == 11:
                        if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (words[3], words[8], words[2]))
                        try:
                            g1f = float(re.search(r'\d{2}\.\d', words[8]).group())*10
                        except:
                            print("parsing error in race_detail - 3")
                            g1f = -1
                    elif len(words) < 9:
                        #print("unexpected line: %s" % line)
                        continue
                    try:
                        s1f = float(re.search(r'\d{2}\.\d', words[3]).group())*10
                    except:
                        s1f = 150
                    try:
                        g3f = float(re.search(r'\d{2}\.\d', words[2]).group())*10
                    except:
                        g3f = 400
                    if s1f < 100 or s1f > 200:
                        s1f = -1
                    if g1f < 100 or g1f > 200:
                        g1f = -1
                    if g3f < 300 or g3f > 500:
                        g3f = -1
                    data[name_list[i]].extend([s1f, g1f, g3f])
                    i += 1
                    
        for k,v in data.items():#.iteritems():
            if len(v) < 5:
                continue
            if k in self.data:
                self.data[k].append(v)
            else:
                self.data[k] = [v]


    def parse_ap_rslt(self, filename):
        in_data = open(filename)
        data = dict()
        course = 900
        while True:
            line = in_data.readline()
            #line = line.encode('utf-8')

            if line is None or len(line) == 0:
                break
            # 제목 : 16년12월25일(일)  제15경주
            date = int(re.search(r'\d{8}', filename).group())
            if re.search(r'주로상태', line) is not None and re.search(r'날.+씨', line) is not None:
                try:
                    humidity = int(re.search(r'(?<=\()[\d ]+(?=\%\))', line).group())
                except:
                    humidity = 10
            # read name
            if re.search(r'산지', line) is not None and (re.search(r'선수명', line) is not None or re.search(r'기수명', line) is not None):
                name_list = []
                while re.search(r'[-─]{5}', line) is None:
                    line = in_data.readline()
                    break
                while True:
                    line = in_data.readline()
                    #line = unicode(line, 'euc-kr').encode('utf-8')
                    if re.search(r'[-─]{5}', line) is not None:
                        res = re.search(r'[-─]{5}', line).group()
                        if DEBUG: print("break: %s" % res)
                        break
                    try:
                        name = re.findall(r'\S+', line)[2].replace('★', '')
                    except:
                        print("except: in %s : %s" % (filename, line))
                    name_list.append(name)
                    data[name] = [date, course, humidity]
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
                    #line = unicode(line, 'euc-kr').encode('utf-8')
                    if re.search(r'[-─]{10}', line) is not None:
                        res = re.search(r'[-─]{10}', line).group()
                        if DEBUG: print("break: %s" % res)
                        break
                    words = re.findall(r'\S+', line)
                    if len(words) < 9:
                        i += 1
                        continue
                    if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (words[3], words[6], words[2]))
                    try:
                        s1f = float(re.search(r'\d{2}\.\d', words[3]).group())*10
                    except:
                        print("parsing error in ap s1f")
                        s1f = -1
                    try:
                        g1f = float(re.search(r'\d{2}\.\d', words[6]).group())*10
                    except:
                        print("parsing error in ap g1f")
                        g1f = -1
                    try:
                        g3f = float(re.search(r'\d{2}\.\d', words[2]).group())*10
                    except:
                        print("parsing error in ap g3f")
                        g3f = -1
                    if s1f < 100 or s1f > 200:
                        s1f = -1
                    if g1f < 100 or g1f > 200:
                        g1f = -1
                    if g3f < 300 or g3f > 500:
                        g3f = -1
                    data[name_list[i]].extend([s1f, g1f, g3f])
                    i += 1
        for k,v in data.items():#.iteritems():
            if len(v) < 5:
                continue
            if k in self.data:
                self.data[k].append(v)
            else:
                self.data[k] = [v]


    def get_data(self, name, date, md=mean_data()):
        name = name.replace('★', '')
        date = int(date)
        res = []
        rs = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        course_list = [900, 1000, 1200, 1300, 1400, 1700]
        means_course = [[], [], []]
        if name in self.data:
            self.data[name] = sorted(self.data[name], key=itemgetter(0))
            for data in self.data[name]:
                if data[0] < date and data[1] in course_list:
                    c = course_list.index(data[1])
                    humidity = int(data[2])
                    if data[3] != -1:
                        value = norm_racescore(data[0]/100%100-1, humidity, data[3], md)
                        rs[3*c+0].append(value)  # s1f
                        means_course[0].append(value / md.race_detail[data[1]][0])
                    if data[4] != -1:
                        value = norm_racescore(data[0]/100%100-1, humidity, data[4], md)
                        rs[3*c+1].append(value)  # g1f
                        means_course[1].append(value / md.race_detail[data[1]][1])
                    if data[5] != -1:
                        value = norm_racescore(data[0]/100%100-1, humidity, data[5], md)
                        rs[3*c+2].append(value)  # g3f
                        means_course[2].append(value / md.race_detail[data[1]][2])
        else:
            print("%s is not in race_detail" % name)
        m_course = [0, 0, 0]
        for i in range(len(rs)):
            if len(rs[i]) == 0:
                res.append(-1)
            else:
                res.append(np.mean(rs[i]))
        if len(means_course[0]) == 0:
            print("can not find race_detail of %s" % name)
        for i in range(len(means_course)):
            if len(means_course[i]) == 0:
                m_course[i] = 1.0
            else:
                m_course[i] = np.mean(means_course[i])
                for j in range(len(means_course[i])):
                    m_course[i] += 0.2*(means_course[i][j] - m_course[i])

        for i in range(len(rs)):
            for j in rs[i]:
                res[i] += 0.2*(j - res[i])
            if res[i] == -1:
                #res[i] = -1
                #res[i] = md.race_detail[course_list[i/3]][i%3]
                res[i] = m_course[i%3] * md.race_detail[course_list[int(i/3)]][i%3]
        return map(lambda x: float(x), res)  # len: 18


def norm_racescore(month, humidity, value, md=mean_data()):
    month = int(month)
    humidity = min(humidity, 20) - 1
    try:
        m = np.array(md.race_score[0])[:,20].mean()
        return value * m / md.race_score[0][month][humidity]
    except KeyError:
        return value


if __name__ == '__main__':
    DEBUG = True
    rd = RaceDetail()
    #for year in range(2007,2017):
    #    filelist1 = glob.glob('../txt/1/ap-check-rslt/ap-check-rslt_1_%d*.txt' % year)
    #    filelist2 = glob.glob('../txt/1/rcresult/rcresult_1_%d*.txt' % year)
    #    for fname in filelist1:
    #        print("processed ap %s" % fname)
    #        rd.parse_ap_rslt(fname)
    #    for fname in filelist2:
    #        print("processed rc in %s" % fname)
    #        rd.parse_race_detail(fname)
    fname1 = '../txt/1/ap-check-rslt/ap-check-rslt_1_20070622.txt'
    fname2 = '../txt/1/rcresult/rcresult_1_20091213.txt'
    rd.parse_ap_rslt(fname1)
    rd.parse_race_detail(fname2)
    print(rd.get_data("등태산", 20110612))
    #joblib.dump(rd, '../data/1_2007_2016_rd.pkl')


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
            if re.search(r'\(제주\)', line) is not None:
                course = int(re.search(r'\d+00*(?=M)', line).group())
            if re.search(r'날씨', line) is not None and re.search(r'주로', line) is not None:
                try:
                    humidity = int(re.search(r'(?<=\()[\d ]+(?=\%\))', line).group())
                except:
                    humidity = 10
            # read name
            if re.search(r'부담중량', line) is not None and re.search(r'마번', line) is not None:
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
                    line = unicode(line, 'euc-kr').encode('utf-8')
                    if re.search(r'[-─]{10}', line) is not None:
                        res = re.search(r'[-─]{10}', line).group()
                        #print("result: %s" % res)
                        break
                    words = re.findall(r'\S+', line)
                    if len(words) == 6:
                        if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (words[3], words[6], words[2]))
                        try:
                            s1f = float(re.search(r'\d{2}\.\d', words[2]).group())*10
                            g1f = float(re.search(r'\d{2}\.\d', words[3]).group())*10
                            g3f = 500
                        except:
                            s1f = 180
                            g1f = 180
                            g3f = 500
                    elif len(words) == 9:
                        if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (words[3], words[6], words[2]))
                        try:
                            s1f = float(re.search(r'\d{2}\.\d', words[3]).group())*10
                            g1f = float(re.search(r'\d{2}\.\d', words[6]).group())*10
                            g3f = float(re.search(r'\d{2}\.\d', words[2]).group())*10
                        except:
                            s1f = 180
                            g1f = 180
                            g3f = 500
                    elif len(words) == 10:
                        if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (words[3], words[7], words[2]))
                        try:
                            s1f = float(re.search(r'\d{2}\.\d', words[3]).group())*10
                            g1f = float(re.search(r'\d{2}\.\d', words[7]).group())*10
                            g3f = float(re.search(r'\d{2}\.\d', words[2]).group())*10
                        except:
                            s1f = 180
                            g1f = 180
                            g3f = 500
                    elif len(words) == 11:
                        if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (words[3], words[8], words[2]))
                        try:
                            s1f = float(re.search(r'\d{2}\.\d', words[3]).group())*10
                            g1f = float(re.search(r'\d{2}\.\d', words[8]).group())*10
                            g3f = float(re.search(r'\d{2}\.\d', words[2]).group())*10
                        except:
                            s1f = 180
                            g1f = 180
                            g3f = 500
                    elif len(words) < 9:
                        #print("unexpected line: %s" % line)
                        continue
                    if s1f < 100 or s1f > 230:
                        s1f = 180
                    if g1f < 100 or g1f > 230:
                        g1f = 180
                    if g3f < 300 or g3f > 600:
                        g3f = 500
                    data[name_list[i]].extend([s1f, g1f, g3f])
                    i += 1
                    
        for k,v in data.iteritems():
            if len(v) < 5:
                continue
            if self.data.has_key(k):
                self.data[k].append(v)
            else:
                self.data[k] = [v]



    def get_data(self, name, date, md=mean_data()):
        res = []
        rs = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        course_list = [300, 400, 800, 900, 1000, 1200]

        for c in range(len(course_list)):
            try:
                for data in self.data[name]:
                    if data[0] < date and data[1] == course_list[c]:
                        humidity = date[2]
                        s1f = norm_racescore(data[0]/100%100, humidity, data[3], md)
                        g1f = norm_racescore(data[0]/100%100, humidity, data[4], md)
                        g3f = norm_racescore(data[0]/100%100, humidity, data[5], md)
                        rs[3*c+0].append(s1f)
                        rs[3*c+1].append(g1f)
                        rs[3*c+2].append(g3f)
            except:
                continue
        for i in range(len(rs)):
            if len(rs[i]) == 0:
                res.append(md.race_detail[course_list[i/3]][i%3])
            else:
                res.append(np.mean(rs[i]))
        for i in range(len(rs)):
            rs[i].reverse()
            for j in rs[i]:
                res[i] += 0.1*(j - res[i])
        return map(lambda x: int(x), res)  # len: 18


def norm_racescore(month, humidity, value, md=mean_data()):
    humidity = min(humidity, 20) - 1
    try:
        return value * np.array(md.race_score[0])[:,20].mean() / md.race_score[0][month][humidity]
    except KeyError:
        return value


if __name__ == '__main__':
    DEBUG = True
    rd = RaceDetail()
    #for year in range(2007,2017):
    #    filelist1 = glob.glob('../txt/2/ap-check-rslt/ap-check-rslt_2_%d*.txt' % year)
    #    filelist2 = glob.glob('../txt/2/rcresult/rcresult_2_%d*.txt' % year)
    #    for fname in filelist1:
    #        print("processed ap %s" % fname)
    #        rd.parse_ap_rslt(fname)
    #    for fname in filelist2:
    #        print("processed rc in %s" % fname)
    #        rd.parse_race_detail(fname)
    fname2 = '../txt/2/rcresult/rcresult_2_20091213.txt'
    rd.parse_race_detail(fname2)
    print(rd.get_data("등태산", 20110612))
    #joblib.dump(rd, '../data/2_2007_2016_rd.pkl')


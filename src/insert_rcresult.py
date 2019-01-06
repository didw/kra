# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os.path
import glob
from sklearn.externals import joblib
from operator import itemgetter

class _RegExLib:
    """Set up regular expressions"""
    # use https://regexper.com to visualise these if required
    _reg_title = re.compile(r'제목 +:')
    _reg_rcno = re.compile(r'(?<=제 *)\d(?=경주)')
    _reg_seoul = re.compile(r'\(서울\)')
    _reg_course = re.compile(r'\d{4}(?=M)')
    _reg_rc_grade = re.compile(r'(?<=M *)[가-힣\d]+(?= )')
    _reg_rc_type = re.compile(r'(?<= )[가-힣A-Z]+(?=경주명)')
    _reg_rc_name = re.compile(r'(?<=경주명 *: *)[가-힣]+')
    _reg_rc_cond = re.compile(r'(?<=경주조건\s*:\s+)\S+')
    _reg_rc_age = re.compile(r'(?<=경주조건\s*:\s+\S+\s+)\S+')
    _reg_rc_weather = re.compile(r'(?<=날씨:)\S+')
    _reg_course_cond = re.compile(r'(?<=주로:)\S+')
    _reg_humidity = re.compile(r'(?<=\()\d+(?=\%)')
    _reg_win_prize = re.compile(r'(?<=착순상금:\s+).*')
    _reg_add_prize = re.compile(r'(?<=부가상금:\s+).*')
    _reg_dash = re.compile(r'[─-]+')

    __slots__ = ['title', 'rcno']

    def __init__(self, line):
        # check whether line has a positive match with all of the regular expressions
        self.title = self._reg_title.match(line)
        self.rcno = self._reg_rcno.search(line)


def parse_race_detail(filename):
    posts = {}
    posts["rc_date"]=os.path.basename(filename)[-12:-4]
    in_file = open(filename)
    line = next(in_file)
    data = dict()
    while True:
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
    line = next(in_file)


def parse_ap_rslt(filename):
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
        if k in data:
            data[k].append(v)
        else:
            data[k] = [v]


if __name__ == '__main__':
    DEBUG = True
    fname1 = '../txt/1/ap-check-rslt/ap-check-rslt_1_20070622.txt'
    fname2 = '../txt/1/rcresult/rcresult_1_20091213.txt'
    parse_ap_rslt(fname1)
    parse_race_detail(fname2)


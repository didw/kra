# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os.path
from urllib2 import urlopen
import get_detail_data as gdd
from bs4 import BeautifulSoup
from mean_data import mean_data
from sklearn.externals import joblib
from get_race_detail import RaceDetail
import get_weekly_clinic as wc
import get_jockey as gj
import get_trainer as gt
import get_lineage as gl
import glob
import time
import cPickle, gzip

NEXT = re.compile(r'마 체 중|단승식|복승식|매출액')
WORD = re.compile(r"[^\s]+")
DEBUG = False

def get_hr_days(name, date_i):
    date = datetime.datetime(date_i/10000, date_i%10000/100, date_i%100)
    while True:
        date = date + datetime.timedelta(days=-1)
        date_s = int("%d%02d%02d" % (date.year, date.month, date.day))
        filename = '../txt/1/horse/horse_1_%s.txt' % date_s
        if os.path.isfile(filename):
            break
    f_input = open(filename)
    while True:
        line = f_input.readline()
        if not line:
            break
        line = unicode(line, 'euc-kr').encode('utf-8')
        hrname = re.search(r'[가-힣]+', line).group()
        if name == hrname:
            birth = re.search(unicode(r'\d{4}/\d{2}/\d{2}', 'utf-8').encode('utf-8'), line).group()
            return (date - datetime.datetime(int(birth[:4]), int(birth[5:7]), int(birth[8:]))).days

d1 = 0
d2 = 0
d3 = 0
d4 = 0
d5 = 0
d11, d12, d13, d14 = 0,0,0,0

def parse_txt_race(filename, race_record):
    global d1, d2, d3, d4, d5
    global d11, d12, d13, d14
    data = []
    input_file = open(filename)
    while True:
        # skip header
        humidity = 0
        read_done = False
        hr_num = [0, 0]
        rcno = -1
        course = ''
        kind = ''
        hrname = ''
        date = int(re.search(r'\d{8}', filename).group())
        month = date/100%100
        t1 = time.time()
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if len(line) == 0:
                read_done = True
                break
            if re.search(unicode(r'제목', 'utf-8').encode('utf-8'), line) is not None:
                rcno = int(re.search(unicode(r'\d+(?=경주)', 'utf-8').encode('utf-8'), line).group())
            if re.search(unicode(r'경주명', 'utf-8').encode('utf-8'), line) is not None:
                if DEBUG: print("%s" % line)
                course = int(re.search(unicode(r'\d+(?=M)', 'utf-8').encode('utf-8'), line).group())
                kind = re.search(unicode(r'M.+\d', 'utf-8').encode('utf-8'), line)
                if kind is None:
                    kind = 0
                else:
                    kind = int(kind.group()[-1])
            if re.search(unicode(r'경주조건', 'utf-8').encode('utf-8'), line) is not None:
                if DEBUG: print("%s" % line)
                if re.search(unicode(r'불량', 'utf-8').encode('utf-8'), line) is not None:
                    humidity = 25
                else:
                    humidity = re.search(unicode(r'\d+(?=%\))', 'utf-8').encode('utf-8'), line)
                    if humidity is None:
                        humidity = '10'
                    else:
                        humidity = humidity.group()
            if re.search(unicode(r'기수명|선수명', 'utf-8').encode('utf-8'), line) is not None:
                break
        if read_done:
            break

        t2 = time.time()
        # 순위 마번    마    명      산지   성별 연령 부담중량 기수명 조교사   마주명           레이팅
        cnt = 0
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if NEXT.search(line) is not None:
                break
            if re.search(unicode(r'[^\s]+', 'utf-8').encode('utf-8'), line[:5]) is None:
                continue
            # 1위와 2위 번호 가져오기
            if hr_num[0] == 0:
                hr_num[0] = re.search(unicode(r'\s*\d\s+\d+', 'utf-8').encode('utf-8'), line[:10]).group().split()[1]
            elif hr_num[1] == 0:
                hr_num[1] = re.search(unicode(r'\s*\d\s+\d+', 'utf-8').encode('utf-8'), line[:10]).group().split()[1]
            hr_num[0] = int(hr_num[0])
            hr_num[1] = int(hr_num[1])
            if hr_num[0] > hr_num[1]:
                tmp = hr_num[0]
                hr_num[0] = hr_num[1]
                hr_num[1] = tmp

            words = WORD.findall(line)
            hrname = words[2]
            t11 = time.time()
            dbudam = gdd.get_dbudam(1, date, int(rcno), hrname)
            t12 = time.time()
            drweight = gdd.get_drweight(1, date, int(rcno), hrname)
            t13 = time.time()
            lastday = gdd.get_lastday(1, date, int(rcno), hrname)
            t14 = time.time()
            hr_days = get_hr_days(hrname, date)
            t15 = time.time()
            d11 += t12-t11
            d12 += t13-t12
            d13 += t14-t13
            d14 += t15-t14

            if len(words) < 10:
                print("something wrong..", filename, words)
            adata = [course, humidity, kind, dbudam, drweight, lastday, hr_days]
            for i in range(1, 10):
                adata.append(words[i])
            data.append(adata)
            cnt += 1

        t3 = time.time()
        # 순위 마번    마      명    마 체 중 기  록  위  차 S1F-1C-2C-3C-4C-G1F
        idx = 0
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if NEXT.search(line) is not None:
                break
            if re.search(unicode(r'[^\s]+', 'utf-8').encode('utf-8'), line[:5]) is None:
                continue
            adata = []
            adata.append(re.search(unicode(r'\d+(?=\()', 'utf-8').encode('utf-8'), line).group())
            adata.append(re.search(unicode(r'[-\d]+(?=\))', 'utf-8').encode('utf-8'), line).group())
            rctime = re.search(unicode(r'\d+:\d+\.\d', 'utf-8').encode('utf-8'), line).group()
            rctime = int(rctime[0])*600 + int(rctime[2:4])*10 + int(rctime[5])
            adata.append(rctime)
            data[-cnt+idx].extend(adata)
            idx += 1

        t4 = time.time()
        # 단승식, 연승식 데이터 가져오기
        # 순위 마번    G-3Ｆ   S-1F  １코너  ２코너  ３코너  ４코너    G-1F  단승식 연승식
        idx = 0
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if NEXT.search(line) is not None:
                break
            if re.search(unicode(r'[^\s]+', 'utf-8').encode('utf-8'), line[:5]) is None:
                continue
            adata = []
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
            adata.append(s1f)
            adata.append(g1f)
            adata.append(g3f)
            data[-cnt+idx].extend(adata)
            idx += 1

        t5 = time.time()
        # 복승식 rating 가져오기
        #   1- 2   949.3  2- 9  1629.8  4- 5   282.5  5-15     0.0  8- 9   519.3 11-12    18.9
        exp = "%d-%2d" % (hr_num[0], hr_num[1])
        get_rate = False
        price = 0
        rating = 0
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if re.search(unicode(r'매출액', 'utf-8').encode('utf-8'), line) is not None:
                break
            parse_line = re.search(unicode(r'%s\s+\d+[.]\d' % exp, 'utf-8').encode('utf-8'), line)

            if parse_line is not None:
                rating = parse_line.group().split('-')[1].split()[1]
                get_rate = True
                break

        for _ in range(300):
            if price != 0:
                break
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if re.search(unicode(r'매출액', 'utf-8').encode('utf-8'), line) is not None:
                break

        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if DEBUG: print("line1: %s" % line)
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:10]) is not None:
                break
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if DEBUG: print("line2: %s" % line)
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                break
            res = re.search(r'(?<=삼쌍:).+', line)
            if res is not None:
                break
        for i in range(cnt):
            data[-cnt + i].extend([cnt])
            data[-cnt + i].extend([rcno])
            data[-cnt + i].extend([month])
        t6 = time.time()
        d1 += t2-t1
        d2 += t3-t2
        d3 += t4-t3
        d4 += t5-t4
        d5 += t6-t5
        # columns:  course, humidity, kind, dbudam, drweight, lastday, hr_days, idx, hrname, cntry, 
        #           gender, age, budam, kockey, trainer, owner, weight, dweight, rctime, s1f, 
        #           g1f, g3f, cnt, rcno, month
    for line in data:
        name = line[8]
        course = int(line[0])
        del line[8]
        del line[0]
        if not name in race_record:
            race_record[name] = {}
        try:
            race_record[name][course].append(line)
        except KeyError:
            race_record[name][course] = [line]
        print("name: %s, course: %d is added => len: %d" % (str(name), course, len(race_record[name][course])))
    print("d1: %d, d2: %d, d3: %d, d4: %d, d5: %d" % (d1, d2, d3, d4, d5))
    print("d11: %d, d12: %d, d13: %d, d14: %d" % (d11, d12, d13, d14))


class RaceRecord:
    def __init__(self):
        self.data = {}
        self.cur_file = 0

    def get_all_record(self):
        flist = glob.glob('../txt/1/rcresult/rcresult_1_2*.txt')
        for fname in sorted(flist):
            if int(fname[-12:-4]) < self.cur_file:
                print("%s is already loaded, pass" % fname)
                continue
            print("%s is processing.." % fname)
            parse_txt_race(fname, self.data)
            self.cur_file = int(fname[-12:-4])
            serialized = cPickle.dumps(self.__dict__)
            with gzip.open('../data/race_record.gz', 'wb') as f:
                f.write(serialized)


if __name__ == '__main__':
    race_record = RaceRecord()
    if os.path.exists('../data/race_record.gz'):
        with gzip.open('../data/race_record.gz', 'rb') as f:
            tmp_dict = cPickle.loads(f.read())
            race_record.__dict__.update(tmp_dict)
    race_record.get_all_record()
    print(race_record)


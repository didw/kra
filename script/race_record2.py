# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os
import get_detail_data as gdd
from bs4 import BeautifulSoup
import get_weekly_clinic as wc
import get_lineage as gl
import glob
import time
import cPickle, gzip
import multiprocessing as mp
from multiprocessing import Value
import Queue


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


def parse_ap_rslt(filename):
    data = []
    input_file = open(filename)
    course = 900
    while True:
        line = input_file.readline()
        line = unicode(line, 'euc-kr').encode('utf-8')
        if line is None or len(line) == 0:
            break
        # 제목 : 16년12월25일(일)  제15경주
        date = int(re.search(r'\d{8}', filename).group())
        month = date/100%100
        kind = 0
        if re.search(r'주로상태', line) is not None and re.search(r'날.+씨', line) is not None:
            try:
                humidity = int(re.search(r'(?<=\()[\d ]+(?=\%\))', line).group())
            except:
                humidity = 10
        # read name
        if re.search(r'산지', line) is not None and (re.search(r'선수명', line) is not None or re.search(r'기수명', line) is not None):
            name_list = []
            while re.search(r'[-─]{5}', line) is None:
                line = input_file.readline()
                break
            while True:
                line = input_file.readline()
                line = unicode(line, 'euc-kr').encode('utf-8')
                if re.search(r'[-─]{5}', line) is not None:
                    res = re.search(r'[-─]{5}', line).group()
                    if DEBUG: print("break: %s" % res)
                    break
                if re.search(unicode(r'[^\s]+', 'utf-8').encode('utf-8'), line[:5]) is None:
                    continue

                words = WORD.findall(line)
                hrname = words[2]
                dbudam = 0
                drweight = 0
                lastday = 30
                hr_days = get_hr_days(hrname, date)
                if len(words) < 9:
                    print("something wrong..", filename, words)
                    continue
                adata = [course, humidity, kind, dbudam, drweight, lastday, hr_days]
                adata.append(int(words[1]))
                adata.append(words[2])
                adata.append(words[3])
                adata.append(words[4])
                adata.append(int(words[5]))
                adata.append(float(words[6]))
                adata.append(words[7])
                adata.append(words[8])
                data.append(adata)
                cnt += 1

        # read score
        if re.search(r'S-1F', line) is not None:
            while re.search(r'[-─]{10}', line) is None:
                line = input_file.readline()
                break
            i = 0
            while True:
                line = input_file.readline()
                line = unicode(line, 'euc-kr').encode('utf-8')
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
    for k,v in data.iteritems():
        if len(v) < 5:
            continue
        if k in self.data:
            self.data[k].append(v)
        else:
            self.data[k] = [v]


def get_data(fname_queue, data_queue, filename_queue):
    fname = fname_queue.get(True, 10)
    print("%s is processing.."%fname)
    data = parse_txt_race(fname)
    date = int(re.search(r'\d{8}', fname).group())
    jangu_clinic = wc.parse_hr_clinic(datetime.date(date/10000, date%10000/100, date%100))
    for i in range(len(data)):
        data[i].extend(wc.get_jangu_clinic(jangu_clinic, data[i][8]))
    data_queue.put(data)
    filename_queue.put(fname)
    print("%s is finished.."%fname)


def update_race_record(data, race_record):
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
        #print("name: %s, course: %d is added => len: %d" % (str(name), course, len(race_record[name][course])))
    #print("d1: %d, d2: %d, d3: %d, d4: %d, d5: %d" % (d1, d2, d3, d4, d5))
    #print("d11: %d, d12: %d, d13: %d, d14: %d" % (d11, d12, d13, d14))


class RaceRecord:
    def __init__(self):
        self.data = {}
        self.cur_file = 0

    def save_data(self, data_queue, filename_queue, worker_num, saving_mode):
        saving_mode.value = 1
        data = data_queue.get(True, 10)
        update_race_record(data, self.data)
        fname = filename_queue.get(True, 10)
        self.cur_file = int(fname[-12:-4])
        if worker_num % 10 == 0:
            print("Saving data...")
            serialized = cPickle.dumps(self.__dict__)
            with gzip.open('../data/race_record.gz', 'wb') as f:
                f.write(serialized)
            print("Done")
        saving_mode.value = 0

    def get_all_record(self):
        flist = glob.glob('../txt/1/rcresult/rcresult_1_2*.txt')
        file_queue = mp.Queue()
        filename_queue = mp.Queue()
        data_queue = mp.Queue()
        for fname in sorted(flist):
            if int(fname[-12:-4]) <= self.cur_file:
                print("%s is already loaded, pass" % fname)
                continue
            file_queue.put(fname)

        worker_num = file_queue.qsize()
        PROCESS_NUM = 12
        saving_mode = Value('i', 0)
        while True:
            print("Processing: %d/%d" % (file_queue.qsize(), worker_num))
            while worker_num < file_queue.qsize() + PROCESS_NUM and file_queue.qsize() > 0:
                print("run process %d" % (worker_num - file_queue.qsize()))
                proc = mp.Process(target=get_data, args=(file_queue, data_queue, filename_queue))
                proc.start()
                time.sleep(5)
            time.sleep(1)
            if saving_mode.value == 0 and data_queue.qsize() > 0:
                try:
                    proc = mp.Process(target=self.save_data, args=(data_queue, filename_queue, worker_num, saving_mode))
                    proc.start()
                    worker_num -= 1
                except Queue.Empty:
                    print("queue empty.. nothing to get data %d" % filename_queue.qsize())
            if worker_num == 0:
                print("job is finished")
                break

if __name__ == '__main__':
    race_record = RaceRecord()
    if os.path.exists('../data/race_record.gz'):
        with gzip.open('../data/race_record.gz', 'rb') as f:
            tmp_dict = cPickle.loads(f.read())
            race_record.__dict__.update(tmp_dict)
    race_record.get_all_record()
    print(race_record)


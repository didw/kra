# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os
import get_detail_data as gdd
from bs4 import BeautifulSoup
import get_weekly_clinic as wc
import glob
import time
import cPickle, gzip
import multiprocessing as mp
from multiprocessing import Value
import Queue


NEXT_AP = re.compile(r'마체중|３코너|4화롱|4펄롱')
NEXT_RC = re.compile(r'마 체 중|단승식|복승식')
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
    return 1000


def parse_txt_race(filename):
    data = []
    input_file = open(filename)
    while True:
        # skip header
        humidity = 0
        read_done = False
        rcno = -1
        course = ''
        kind = ''
        hrname = ''
        date = int(re.search(r'\d{8}', filename).group())
        month = date/100%100
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
                    humidity = 20
                else:
                    humidity = re.search(unicode(r'\d+(?=%\))', 'utf-8').encode('utf-8'), line)
                    if humidity is None:
                        humidity = 10
                    else:
                        humidity = int(humidity.group())
            if re.search(unicode(r'기수명|선수명', 'utf-8').encode('utf-8'), line) is not None:
                break
        if read_done:
            break

        # 순위 마번    마    명      산지   성별 연령 부담중량 기수명 조교사   마주명           레이팅
        cnt = 0
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if NEXT_RC.search(line) is not None:
                break
            if re.search(unicode(r'[^\s]+', 'utf-8').encode('utf-8'), line[:5]) is None:
                continue

            words = WORD.findall(line)
            hrname = words[2]
            dbudam = gdd.get_dbudam(1, date, int(rcno), hrname)
            drweight = gdd.get_drweight(1, date, int(rcno), hrname)
            lastday = gdd.get_lastday(1, date, int(rcno), hrname)
            hr_days = get_hr_days(hrname, date)

            if len(words) < 10:
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
            adata.append(words[9])
            data.append(adata)
            assert len(data[-1]) == 16
            cnt += 1

        # 순위 마번    마      명    마 체 중 기  록  위  차 S1F-1C-2C-3C-4C-G1F
        idx = 0
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if NEXT_RC.search(line) is not None:
                break
            if re.search(unicode(r'[^\s]+', 'utf-8').encode('utf-8'), line[:5]) is None:
                continue
            adata = []
            adata.append(int(re.search(unicode(r'\d+(?=\()', 'utf-8').encode('utf-8'), line).group()))
            adata.append(int(re.search(unicode(r'[-\d]+(?=\))', 'utf-8').encode('utf-8'), line).group()))
            rctime = re.search(unicode(r'\d+:\d+\.\d', 'utf-8').encode('utf-8'), line).group()
            rctime = int(rctime[0])*600 + int(rctime[2:4])*10 + int(rctime[5])
            adata.append(rctime)
            data[-cnt+idx].extend(adata)
            assert len(data[-cnt+idx]) == 19
            idx += 1

        # 순위 마번    G-3Ｆ   S-1F  １코너  ２코너  ３코너  ４코너    G-1F  단승식 연승식
        idx = 0
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if NEXT_RC.search(line) is not None:
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
            assert len(data[-cnt+idx]) == 22
            idx += 1

        for i in range(cnt):
            data[-cnt + i].extend([cnt])
            data[-cnt + i].extend([rcno])
            data[-cnt + i].extend([month])
            data[-cnt + i].extend([date])
            assert len(data[-cnt+idx]) == 26
        assert len(data[-1]) == 26
        # columns:  course, humidity, kind, dbudam, drweight, lastday, hr_days, idx, hrname, cntry, 
        #           gender, age, budam, jockey, trainer, owner, weight, dweight, rctime, s1f
        #           g1f, g3f, cnt, rcno, month, date
    return data


def parse_ap_rslt(filename):
    data = []
    input_file = open(filename)
    while True:
        # skip header
        humidity = 0
        read_done = False
        rcno = -1
        course = 900
        kind = 0
        hrname = ''
        date = int(re.search(r'\d{8}', filename).group())
        month = date/100%100
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if len(line) == 0:
                read_done = True
                break
            if re.search(unicode(r'제목', 'utf-8').encode('utf-8'), line) is not None:
                rcno = int(re.search(unicode(r'\d+(?=경주)', 'utf-8').encode('utf-8'), line).group())
            if re.search(unicode(r'주로상태', 'utf-8').encode('utf-8'), line) is not None:
                if DEBUG: print("%s" % line)
                if re.search(unicode(r'불량', 'utf-8').encode('utf-8'), line) is not None:
                    humidity = 20
                else:
                    humidity = re.search(unicode(r'\d+(?=%\))', 'utf-8').encode('utf-8'), line)
                    if humidity is None:
                        humidity = 10
                    else:
                        humidity = int(humidity.group())
            if re.search(unicode(r'기수명|선수명', 'utf-8').encode('utf-8'), line) is not None:
                break
        if read_done:
            break

        # 순위  마번  마    명         산지  성  연령   부  담  기수명    조교사명
        cnt = 0
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if NEXT_AP.search(line) is not None:
                break
            if re.search(unicode(r'[^\s]+', 'utf-8').encode('utf-8'), line[:6]) is None:
                continue

            words = WORD.findall(line)
            if words[0][0] == '9' and len(words[0])==2:
                continue
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
            adata.append(np.sum(map(float, words[6].split('+'))))
            adata.append(words[7])
            adata.append(words[8])
            adata.append("NoOwner")
            data.append(adata)
            assert len(data[-1]) == 16
            cnt += 1

        # 순위 마번  마    명        마체중 기  록  도착차    판정 불합격사유   검사사유
        idx = 0
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if NEXT_AP.search(line) is not None:
                break
            if re.search(unicode(r'[^\s]+', 'utf-8').encode('utf-8'), line[:6]) is None:
                continue
            adata = []
            words = WORD.findall(line)
            if words[0][0] == '9' and len(words[0])==2:
                continue
            adata.append(int(words[3]))
            adata.append(0)
            rctime = re.search(unicode(r'\d+:\d+\.\d', 'utf-8').encode('utf-8'), line).group()
            rctime = int(rctime[0])*600 + int(rctime[2:4])*10 + int(rctime[5])
            adata.append(rctime)
            data[-cnt+idx].extend(adata)
            assert len(data[-cnt+idx]) == 19
            idx += 1

        # 순위  마번    G-3Ｆ    S-1F   ３코너   ４코너     G-1F  S1F-1C-2C-3C-4C-G1F
        idx = 0
        for _ in range(300):
            END_CHK = re.compile(r'4화롱|4펄롱')
            if END_CHK.search(line) is not None:
                for idx in range(cnt):
                    data[-cnt+idx].extend([150, 150, 400])
                break
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if NEXT_AP.search(line) is not None:
                break
            if re.search(unicode(r'[^\s]+', 'utf-8').encode('utf-8'), line[:6]) is None:
                continue
            adata = []
            words = re.findall(r'\S+', line)
            if words[0][0] == '9' and len(words[0])==2:
                continue
            s1f, g1f, g3f = -1, -1, -1
            if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (words[3], words[6], words[2]))
            try:
                g1f = float(re.search(r'\d{2}\.\d', words[6]).group())*10
            except:
                g1f = 150
            try:
                s1f = float(re.search(r'\d{2}\.\d', words[3]).group())*10
            except:
                s1f = 150
            try:
                g3f = float(re.search(r'\d{2}\.\d', words[2]).group())*10
            except:
                g3f = 400
            if s1f < 100 or s1f > 200:
                s1f = 150
            if g1f < 100 or g1f > 200:
                g1f = 150
            if g3f < 300 or g3f > 500:
                g3f = 400
            adata.append(s1f)
            adata.append(g1f)
            adata.append(g3f)
            data[-cnt+idx].extend(adata)
            assert len(data[-cnt+idx]) == 22
            idx += 1

        for i in range(cnt):
            data[-cnt + i].extend([cnt])
            data[-cnt + i].extend([rcno])
            data[-cnt + i].extend([month])
            data[-cnt + i].extend([date])
            assert len(data[-cnt + i]) == 26
        assert len(data[-1]) == 26
        # columns:  course, humidity, kind, dbudam, drweight, lastday, hr_days, idx, hrname, cntry, 
        #           gender, age, budam, jockey, trainer, owner, weight, dweight, rctime, s1f, 
        #           g1f, g3f, cnt, rcno, month, date
    return data


def get_data(function_name, fname_queue, data_queue, filename_queue):
    fname = fname_queue.get(True, 10)
    filename_queue.put(fname)
    print("%s is processing.."%fname)
    data = function_name(fname)
    date = int(re.search(r'\d{8}', fname).group())
    jangu_clinic = wc.parse_hr_clinic(datetime.date(date/10000, date%10000/100, date%100))
    for i in range(len(data)):
        data[i].extend(wc.get_jangu_clinic(jangu_clinic, data[i][8]))
    data_queue.put(data)


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
    # idx:  humidity, kind, dbudam, drweight, lastday, hr_days, idx, cntry, gender, age, 
    #       budam, jockey, trainer, owner, weight, dweight, rctime, s1f, g1f, g3f, 
    #       cnt, rcno, month, date, jc1 ~ jc81


class RaceRecord:
    def __init__(self):
        self.data = {}
        self.cur_rc_file = 0
        self.cur_ap_file = 0

    def save_data(self, data_queue, filename_queue, worker_num, saving_mode):
        saving_mode.value = 1
        data = data_queue.get(True, 10)
        update_race_record(data, self.data)
        print("len(self.data): %d" % len(self.data.keys()))
        fname = filename_queue.get(True, 10)
        if "rcresult" in fname:
            self.cur_rc_file = int(fname[-12:-4])
        if "ap" in fname:
            self.cur_ap_file = int(fname[-12:-4])
        if worker_num % 10 == 1:
            if "rcresult" in fname:
                print("saved rc at %d" % self.cur_rc_file)
            if "ap" in fname:
                print("saved ap at %d" % self.cur_ap_file)
            print("Saving data...")
            serialized = cPickle.dumps(self.__dict__)
            with gzip.open('../data/race_record.gz', 'wb') as f:
                f.write(serialized)
            print("Done")
        saving_mode.value = 0

    def get_race_record(self):
        flist = sorted(glob.glob('../txt/1/rcresult/rcresult_1_20*.txt'))
        file_queue = mp.Queue()
        filename_queue = mp.Queue()
        data_queue = mp.Queue()
        for fname in sorted(flist):
            if int(fname[-12:-4]) <= self.cur_rc_file:
                print("%s is already loaded, pass" % fname)
                continue
            file_queue.put(fname)

        worker_num = file_queue.qsize()
        PROCESS_NUM = 5
        saving_mode = Value('i', 0)
        while True:
            print("Processing: %d/%d" % (file_queue.qsize(), worker_num))
            time.sleep(0.5)
            while worker_num < file_queue.qsize() + PROCESS_NUM and file_queue.qsize() > 0:
                print("run process %d" % (worker_num - file_queue.qsize()))
                proc = mp.Process(target=get_data, args=(parse_txt_race, file_queue, data_queue, filename_queue))
                proc.start()
                if file_queue.qsize() < 20:
                    time.sleep(5)
                else:
                    time.sleep(2)
            if saving_mode.value == 0 and data_queue.qsize() > 0:
                try:
                    #proc = mp.Process(target=self.save_data, args=(data_queue, filename_queue, worker_num, saving_mode))
                    #proc.start()
                    self.save_data(data_queue, filename_queue, worker_num, saving_mode)
                    #time.sleep(2)
                    worker_num -= 1
                except Queue.Empty:
                    print("queue empty.. nothing to get data %d" % filename_queue.qsize())
            if worker_num == 0:
                print("job is finished")
                break

    def get_ap_record(self):
        flist = sorted(glob.glob('../txt/1/ap-check-rslt/ap-check-rslt_1_20*.txt'))
        file_queue = mp.Queue()
        filename_queue = mp.Queue()
        data_queue = mp.Queue()
        for fname in sorted(flist):
            if int(fname[-12:-4]) <= self.cur_ap_file:
                print("%s is already loaded, pass" % fname)
                continue
            file_queue.put(fname)

        worker_num = file_queue.qsize()
        PROCESS_NUM = 5
        saving_mode = Value('i', 0)
        while True:
            print("Processing: %d/%d" % (file_queue.qsize(), worker_num))
            time.sleep(0.5)
            while worker_num < file_queue.qsize() + PROCESS_NUM and file_queue.qsize() > 0:
                print("run process %d" % (worker_num - file_queue.qsize()))
                proc = mp.Process(target=get_data, args=(parse_ap_rslt, file_queue, data_queue, filename_queue))
                proc.start()
                if file_queue.qsize() < 20:
                    time.sleep(5)
                else:
                    time.sleep(2)
            if saving_mode.value == 0 and data_queue.qsize() > 0:
                try:
                    #proc = mp.Process(target=self.save_data, args=(data_queue, filename_queue, worker_num, saving_mode))
                    #proc.start()
                    self.save_data(data_queue, filename_queue, worker_num, saving_mode)
                    #time.sleep(1)
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
    race_record.get_race_record()
    race_record.get_ap_record()
    print(race_record)


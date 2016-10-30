# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os.path


NEXT = re.compile(unicode(r'마 체 중|단승식', 'utf-8').encode('utf-8'))
WORD = re.compile(r"[^\s]+")


def parse_txt_race(input_file):
    data = []
    while True:
        # skip header
        humidity = 0
        read_done = False
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if len(line) == 0:
                read_done = True
                break
            if re.search(unicode(r'경주명', 'utf-8').encode('utf-8'), line) is not None:
                course = re.search(unicode(r'\d+(?=M)', 'utf-8').encode('utf-8'), line).group()
            if re.search(unicode(r'경주조건', 'utf-8').encode('utf-8'), line) is not None:
                if re.search(unicode(r'불량', 'utf-8').encode('utf-8'), line) is not None:
                    humidity = 25
                else:
                    humidity = re.search(unicode(r'\d+(?=%\))', 'utf-8').encode('utf-8'), line).group()
            if re.search(unicode(r'기수명', 'utf-8').encode('utf-8'), line) is not None:
                break
        if read_done:
            break
        # 순위 마번    마    명      산지   성별 연령 부담중량 기수명 조교사   마주명           레이팅
        cnt = 0
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if line[0] == '-':
                continue
            if NEXT.search(line) is not None:
                break
            if re.search(unicode(r'[^\s]+', 'utf-8').encode('utf-8'), line[:5]) is None:
                continue
            words = WORD.findall(line)
            adata = [course, humidity]
            for i in range(10):
                adata.append(words[i])
            data.append(adata)
            cnt += 1

        # 순위 마번    마      명    마 체 중 기  록  위  차 S1F-1C-2C-3C-4C-G1F
        idx = 0
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if line[0] == '-':
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
    return data



def get_fname(date, job):
    while True:
        if date.weekday() == 5:
            date = date + datetime.timedelta(days=-2)
        elif date.weekday() == 6:
            date = date + datetime.timedelta(days=-3)
        elif date.weekday() == 3:
            date = date + datetime.timedelta(days=-4)
        date_s = int("%d%02d%02d" % (date.year, date.month, date.day))
        filename = '../txt/%s/%s_1_%s.txt' % (job, job, date_s)
        if os.path.isfile(filename):
            return filename
        return filename
    return -1


# 이름             산지  성별   birth  -    조교사  마주명             -                    -                     총경기, 총1, 총2, 1년경기, 1년1, 1년2,총상금
# 킹메신저          한    수2014/03/08 2국6 18박대흥죽마조합            시에로골드          난초                    1    0    0    1    0    0    3000000                     0
def parse_txt_horse(date, name):
    filename = get_fname(date, "horse")
    #print(filename)
    f_input = open(filename)
    while True:
        line = f_input.readline()
        line = unicode(line, 'euc-kr').encode('utf-8')
        if len(line) == 0:
            break
        if re.search(unicode(name, 'utf-8').encode('utf-8'), line) is not None:
            data = []
            birth = re.search(unicode(r'\d{4}/\d{2}/\d{2}', 'utf-8').encode('utf-8'), line).group()
            #print(datetime.date(int(birth[:4]), int(birth[5:7]), int(birth[8:])))
            data.append((date - datetime.date(int(birth[:4]), int(birth[5:7]), int(birth[8:]))).days)
            participates = re.search(unicode(r'\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s', 'utf-8').encode('utf-8'), line).group().split()
            #print(participates)
            if float(participates[0]) == 0:
                data.append(-1)
                data.append(-1)
            else:
                data.append(float(participates[1])/float(participates[0]))
                data.append(float(participates[2])/float(participates[0]))

            if float(participates[3]) == 0:
                data.append(-1)
                data.append(-1)
            else:
                data.append(float(participates[4])/float(participates[3]))
                data.append(float(participates[5])/float(participates[3]))

            return data
    print("something wrong in parse_txt_horse")
    return [-1,-1,-1,-1]


# 이름  소속 생일        데뷔일  총경기수, 총1, 총2, 1년, 1년1, 1년2
# 김동철491974/11/28371995/07/015252 3706  217  242  166   17   19
def parse_txt_jockey(date, name):
    filename = get_fname(date, "jockey")
    #print(filename)
    f_input = open(filename)
    while True:
        line = f_input.readline()
        line = unicode(line, 'euc-kr').encode('utf-8')
        if len(line) == 0:
            break
        if re.search(unicode(name, 'utf-8').encode('utf-8'), line) is not None:
            data = []
            participates = re.search(unicode(r'(?<=\d{6})[\s\d]+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s', 'utf-8').encode('utf-8'), line).group().split()
            #print(participates)
            if float(participates[0]) == 0:
                data.append(-1)
                data.append(-1)
            else:
                data.append(float(participates[1])/float(participates[0]))
                data.append(float(participates[2])/float(participates[0]))

            if float(participates[3]) == 0:
                data.append(-1)
                data.append(-1)
            else:
                data.append(float(participates[4])/float(participates[3]))
                data.append(float(participates[5])/float(participates[3]))

            return data
    print("something wrong in parse_txt_jockey")
    return [-1,-1,-1,-1]


# 이름  소속 생일        데뷔일  총경기수, 총1, 총2, 1년, 1년1, 1년2
# 곽영효191961/09/24551997/05/283,868  438  394  134   18   13
def parse_txt_trainer(date, name):
    filename = get_fname(date, "trainer")
    #print(filename)
    f_input = open(filename)
    while True:
        line = f_input.readline()
        line = unicode(line, 'euc-kr').encode('utf-8')
        if len(line) == 0:
            break
        if re.search(unicode(name, 'utf-8').encode('utf-8'), line) is not None:
            data = []
            participates = re.search(unicode(r'(?<=/\d\d)[\d,]+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+', 'utf-8').encode('utf-8'),
                                     line).group().replace(',', '').split()
            #print(participates)
            if float(participates[0]) == 0:
                data.append(-1)
                data.append(-1)
            else:
                data.append(float(participates[1])/float(participates[0]))
                data.append(float(participates[2])/float(participates[0]))

            if float(participates[3]) == 0:
                data.append(-1)
                data.append(-1)
            else:
                data.append(float(participates[4])/float(participates[3]))
                data.append(float(participates[5])/float(participates[3]))

            return data
    print("something wrong in parse_txt_trainer")
    return [-1,-1,-1,-1]


def get_data(filename):
    print("race file: %s" % filename)
    date = re.search(unicode(r'\d{8}', 'utf-8').encode('utf-8'), filename).group()
    date = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:]))
    data = parse_txt_race(open(filename))
    for i in range(len(data)):
        #print("race file: %s" % filename)
        data[i].extend(parse_txt_horse(date, data[i][4]))
        data[i].extend(parse_txt_jockey(date, data[i][9]))
        data[i].extend(parse_txt_trainer(date, data[i][10]))
    df = pd.DataFrame(data)
    df.columns = ['course', 'humidity', 'rank', 'idx', 'name', 'cntry', 'gender', 'age', 'budam', 'jockey', 'trainer', 'owner', \
                  'weight', 'dweight', 'rctime', 'hr_days', 'hr_t1', 'hr_t2', 'hr_y1', 'hr_y2', \
                  'jk_t1', 'jk_t2', 'jk_y1', 'jk_y2', 'tr_t1', 'tr_t2', 'tr_y1', 'tr_y2']
    return df


if __name__ == '__main__':
    filename = '../txt/rcresult/rcresult_1_20160213.txt'
    data = get_data(filename)
    print(data)




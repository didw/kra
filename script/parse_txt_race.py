# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os.path


NEXT = re.compile(unicode(r'마 체 중|단승식|복승식|매출액', 'utf-8').encode('utf-8'))
WORD = re.compile(r"[^\s]+")


def parse_txt_race(input_file):
    data = []
    while True:
        # skip header
        humidity = 0
        read_done = False
        hr_num = [0, 0]
        rcno = -1
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if len(line) == 0:
                read_done = True
                break
            if re.search(unicode(r'제목', 'utf-8').encode('utf-8'), line) is not None:
                rcno = re.search(unicode(r'\d+(?=경주)', 'utf-8').encode('utf-8'), line).group()
            if re.search(unicode(r'경주명', 'utf-8').encode('utf-8'), line) is not None:
                course = re.search(unicode(r'\d+(?=M)', 'utf-8').encode('utf-8'), line).group()
            if re.search(unicode(r'경주조건', 'utf-8').encode('utf-8'), line) is not None:
                if re.search(unicode(r'불량', 'utf-8').encode('utf-8'), line) is not None:
                    humidity = 25
                else:
                    humidity = re.search(unicode(r'\d+(?=%\))', 'utf-8').encode('utf-8'), line).group()
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
            assert len(words) >= 10
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
            rating = re.search(unicode(r'\d+[.]\d\s+\d+[.]\d\s*$', 'utf-8').encode('utf-8'), line).group().split()
            adata.append(rating[0])
            adata.append(rating[1])
            data[-cnt+idx].extend(adata)
            idx += 1

        # 복승식 rating 가져오기
        #   1- 2   949.3  2- 9  1629.8  4- 5   282.5  5-15     0.0  8- 9   519.3 11-12    18.9
        exp = "%d-%2d" % (hr_num[0], hr_num[1])
        get_rate = False
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if NEXT.search(line) is not None:
                break
            parse_line = re.search(unicode(r'%s\s+\d+[.]\d' % exp, 'utf-8').encode('utf-8'), line)

            if parse_line is not None:
                rating = parse_line.group().split('-')[1].split()[1]
                for i in range(cnt):
                    data[-cnt+i].extend([rating])
                    data[-cnt+i].extend([cnt])
                    data[-cnt+i].extend([rcno])
                get_rate = True
                break
        assert get_rate
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
            participates = re.search(unicode(r'\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s', 'utf-8').encode('utf-8'), line).group().replace(',', '').split()
            #print(participates)
            if int(participates[0]) == 0:
                data.extend([0, 0, 0, 0, 0])
            else:
                data.append(int(participates[0]))
                data.append(int(participates[1]))
                data.append(int(participates[2]))
                data.append(int(participates[1])*100/int(participates[0]))
                data.append(int(participates[2])*100/int(participates[0]))

            if int(participates[3]) == 0:
                data.extend([0, 0, 0, 0, 0])
            else:
                data.append(int(participates[3]))
                data.append(int(participates[4]))
                data.append(int(participates[5]))
                data.append(int(participates[4])*100/int(participates[3]))
                data.append(int(participates[5])*100/int(participates[3]))

            return data
    print("can not find %s in %s" % (name, filename))
    return [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]


# 이름  소속 생일        데뷔일  총경기수, 총1, 총2, 1년, 1년1, 1년2
# 김동철491974/11/28371995/07/015252 3706  217  242  166   17   19
def parse_txt_jockey(date, name):
    if len(str(name)) > 9:
        #print("name is changed %s -> %s" % (name, name[:6]))
        name = str(name)[:6]
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
            participates = re.search(unicode(r'(?<=[\d\s]{6})[\s\d]+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s', 'utf-8').encode('utf-8'), line).group().replace(',', '').split()
            #print(participates)
            if int(participates[0]) == 0:
                data.extend([0, 0, 0, 0, 0])
            else:
                data.append(int(participates[0]))
                data.append(int(participates[1]))
                data.append(int(participates[2]))
                data.append(int(participates[1])*100/int(participates[0]))
                data.append(int(participates[2])*100/int(participates[0]))

            if int(participates[3]) == 0:
                data.extend([0, 0, 0, 0, 0])
            else:
                data.append(int(participates[3]))
                data.append(int(participates[4]))
                data.append(int(participates[5]))
                data.append(int(participates[4])*100/int(participates[3]))
                data.append(int(participates[5])*100/int(participates[3]))

            return data
    print("can not find %s in %s" % (name, filename))
    return [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]


# 이름  소속 생일        데뷔일  총경기수, 총1, 총2, 1년, 1년1, 1년2
# 곽영효191961/09/24551997/05/283,868  438  394  134   18   13
def parse_txt_trainer(date, name):
    if len(str(name)) > 9:
        #print("name is changed %s -> %s" % (name, name[:6]))
        name = str(name)[:6]
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
            if int(participates[0]) == 0:
                data.extend([0, 0, 0, 0, 0])
            else:
                data.append(int(participates[0]))
                data.append(int(participates[1]))
                data.append(int(participates[2]))
                data.append(int(participates[1])*100/int(participates[0]))
                data.append(int(participates[2])*100/int(participates[0]))

            if int(participates[3]) == 0:
                data.extend([0, 0, 0, 0, 0])
            else:
                data.append(int(participates[3]))
                data.append(int(participates[4]))
                data.append(int(participates[5]))
                data.append(int(participates[4])*100/int(participates[3]))
                data.append(int(participates[5])*100/int(participates[3]))

            return data
    print("can not find %s in %s" % (name, filename))
    return [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]


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
    df.columns = ['course', 'humidity', 'rank', 'idx', 'name', 'cntry', 'gender', 'age', 'budam', 'jockey', 'trainer',
                  'owner', 'weight', 'dweight', 'rctime', 'r1', 'r2', 'r3', 'cnt', 'rcno', 'hr_days', 'hr_nt', 'hr_nt1', 'hr_nt2',
                  'hr_t1', 'hr_t2', 'hr_ny', 'hr_ny1', 'hr_ny2', 'hr_y1', 'hr_y2', 'jk_nt', 'jk_nt1', 'jk_nt2', 'jk_t1',
                  'jk_t2', 'jk_ny', 'jk_ny1', 'jk_ny2', 'jk_y1', 'jk_y2', 'tr_nt', 'tr_nt1', 'tr_nt2', 'tr_t1', 'tr_t2',
                  'tr_ny', 'tr_ny1', 'tr_ny2', 'tr_y1', 'tr_y2']
    return df


if __name__ == '__main__':
    filename = '../txt/rcresult/rcresult_1_20140831.txt'
    data = get_data(filename)
    data = data.dropna()
    print(data)
    print(data[['rctime', 'r1', 'r2', 'r3']])
    print(data['cnt'])
    print(data['rcno'])



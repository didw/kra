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

NEXT = re.compile(unicode(r'마 체 중|단승식|복승식|매출액', 'utf-8').encode('utf-8'))
WORD = re.compile(r"[^\s]+")
DEBUG = False

def parse_txt_race(filename, md=mean_data()):
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
        month = 0
        date = int(re.search(r'\d{8}', filename).group())
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if len(line) == 0:
                read_done = True
                break
            if re.search(unicode(r'제목', 'utf-8').encode('utf-8'), line) is not None:
                rcno = re.search(unicode(r'\d+(?=경주)', 'utf-8').encode('utf-8'), line).group()
                month = re.search(unicode(r'\d+(?=월)', 'utf-8').encode('utf-8'), line).group()
            if re.search(unicode(r'경주명', 'utf-8').encode('utf-8'), line) is not None:
                if DEBUG: print("%s" % line)
                course = re.search(unicode(r'\d+(?=M)', 'utf-8').encode('utf-8'), line).group()
                kind = re.search(unicode(r'M.+\d', 'utf-8').encode('utf-8'), line)
                if kind is None:
                    kind = 0
                else:
                    kind = kind.group()[-1]
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
            dbudam = gdd.get_dbudam(3, date, int(rcno), hrname)
            drweight = gdd.get_drweight(3, date, int(rcno), hrname)
            lastday = gdd.get_lastday(3, date, int(rcno), hrname)
            train_state = gdd.get_train_state(3, date, int(rcno), hrname)
            hr_no = gdd.get_hrno(3, date, int(rcno), hrname)
            race_score = gdd.get_hr_racescore(3, hr_no, date, course, 'File', md)

            assert len(words) >= 10
            adata = [course, humidity, kind, dbudam, drweight, lastday]
            adata.extend(train_state)
            adata.extend(race_score)
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
        price = 0
        rating = 0
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                continue
            if re.search(unicode(r'매출액', 'utf-8').encode('utf-8'), line) is not None:
                price = int(re.search(unicode(r'(?<=단식:)[ ,\d]+', 'utf-8').encode('utf-8'), line).group().replace(',', ''))
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
                price = int(re.search(unicode(r'(?<=단식:)[ ,\d]+', 'utf-8').encode('utf-8'), line).group().replace(',', ''))
                break

        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if DEBUG: print("line1: %s" % line)
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:10]) is not None:
                break
        bokyeon = ['-1', '-1', '-1']
        boksik = '-1'
        ssang = '-1'
        sambok = '-1'
        samssang = '-1'
        for _ in range(300):
            line = input_file.readline()
            line = unicode(line, 'euc-kr').encode('utf-8')
            if DEBUG: print("line2: %s" % line)
            if re.match(unicode(r'[-─]+', 'utf-8').encode('utf-8'), line[:5]) is not None:
                break
            res = re.search(r'(?<= 복:).+(?=4F)', line)
            if res is not None:
                res = res.group().split()
                if len(res) == 2:
                    boksik = res[1]
                elif len(res) == 1:
                    boksik = res[0][6:]
                else:
                    print("not expected.. %s" % line)
            res = re.search(r'(?<= 쌍:).+', line)
            if res is not None:
                res = res.group().split()
                if len(res) >= 2 and re.search(r'\d', res[1][:1]) is not None:
                    ssang = res[1]
                elif len(res) == 1 and len(res[0]) > 6:
                    ssang = res[0][6:]
                else:
                    print("not expected.. %s" % line)
            res = re.search(r'(?<=복연:).+', line)
            if res is not None:
                res = res.group().split()
                if len(res) >= 6 and re.search(r'\d', res[1][:1]) is not None:
                    bokyeon[0] = "%s%s" % (res[0], res[1])
                    bokyeon[1] = "%s%s" % (res[2], res[3])
                    bokyeon[2] = "%s%s" % (res[4], res[5])
                elif len(res) == 3:
                    bokyeon = res
                else:
                    print("can not parsing.. %s in %s" % (line, filename))
            res = re.search(r'(?<=삼복:).+', line)
            if res is not None:
                res = res.group().split()
                if len(res) >= 2 and re.search(r'\d', res[1][:1]) is not None:
                    sambok = res[1]
                elif len(res) == 1 and len(res[0]) > 9:
                    sambok = res[0][9:]
                else:
                    print("not expected.. %s" % line)
            res = re.search(r'(?<=삼쌍:).+', line)
            if res is not None:
                res = res.group().split()
                if len(res) >= 2 and re.search(r'\d', res[1][:1]) is not None:
                    sambok = res[1]
                elif len(res) == 1 and len(res[0]) > 9:
                    samssang = res[0][9:]
                else:
                    print("not expected.. %s" % line)
                break
        if DEBUG:
            print("price: %s" % price)
            print("%d, %s, %s, %s" % (len(bokyeon), bokyeon[0], bokyeon[1], bokyeon[2]))
            print("%s" % boksik)
            print("%s" % ssang)
            print("%s" % sambok)
            print("%s" % samssang)
        for i in range(cnt):
            data[-cnt + i].extend([rating])
            data[-cnt + i].extend([cnt])
            data[-cnt + i].extend([rcno])
            data[-cnt + i].extend([month])
            data[-cnt + i].extend([price])
            data[-cnt + i].extend(bokyeon)
            data[-cnt + i].append(boksik)
            data[-cnt + i].append(ssang)
            data[-cnt + i].append(sambok)
            data[-cnt + i].append(samssang)
        # 쌍, 복연, 삼복, 삼쌍 배당률 가져오기

        assert get_rate
    return data

def get_humidity():
    url = "http://race.kra.co.kr/chulmainfo/trackView.do?Act=02&Sub=10&meet=3"
    response_body = urlopen(url).read()
    line = unicode(response_body, 'euc-kr').encode('utf-8')
    #print("%s" % line)
    p = re.compile(unicode(r'(?<=함수율 <span>: )\d+(?=\%\()', 'utf-8').encode('utf-8'), re.MULTILINE)
    pl = p.search(line)
    res = 10
    if pl is not None:
        res = pl.group()
    return res

def get_fname(date, job):
    while True:
        date = date + datetime.timedelta(days=-1)
        date_s = int("%d%02d%02d" % (date.year, date.month, date.day))
        filename = '../txt/3/%s/%s_3_%s.txt' % (job, job, date_s)
        if os.path.isfile(filename):
            return filename
    return -1


def get_fname_dist(date, rcno):
    while True:
        date_s = int("%d%02d%02d" % (date.year, date.month, date.day))
        filename = '../txt/3/dist_rec/dist_rec_3_%s_%d.txt' % (date_s, rcno)
        if os.path.isfile(filename):
            return filename
        date = date + datetime.timedelta(days=-1)
    return -1


# 이름             산지  성별   birth  -    조교사  마주명             -                    -                     총경기, 총1, 총2, 1년경기, 1년1, 1년2,총상금
# 킹메신저          한    수2014/03/08 2국6 18박대흥죽마조합            시에로골드          난초                    1    0    0    1    0    0    3000000                     0
def parse_txt_horse(date, rcno, name, course, md=mean_data()):
    name = name.replace('★', '')
    filename = get_fname(date, "horse")
    course = int(course)
    #print(filename)
    f_input = open(filename)
    while True:
        line = f_input.readline()
        if not line:
            break
        line = unicode(line, 'euc-kr').encode('utf-8')
        hrname = re.search(r'[가-힣]+', line).group()
        if name == hrname:
            data = []
            birth = re.search(unicode(r'\d{4}/\d{2}/\d{2}', 'utf-8').encode('utf-8'), line).group()
            #print(datetime.date(int(birth[:4]), int(birth[5:7]), int(birth[8:])))
            data.append((date - datetime.date(int(birth[:4]), int(birth[5:7]), int(birth[8:]))).days)
            participates = re.search(unicode(r'\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s', 'utf-8').encode('utf-8'), line).group().replace(',', '').split()
            dist_rec = gdd.get_distance_record(3, name, rcno, date, course)
            #print(participates)
            if int(participates[0]) == 0:
                #data.extend([0, -1, -1, -1, -1])
                data.extend([0] + md.hr_history_total[course][1:])
            else:
                data.append(int(participates[0]))
                data.append(int(participates[1]))
                data.append(int(participates[2]))
                data.append(int(int(participates[1])*100/int(participates[0])))
                data.append(int(int(participates[2])*100/int(participates[0])))

            if int(participates[3]) == 0:
                #data.extend([0, -1, -1, -1, -1])
                data.extend([0] + md.hr_history_year[course][1:])
            else:
                data.append(int(participates[3]))
                data.append(int(participates[4]))
                data.append(int(participates[5]))
                data.append(int(int(participates[4])*100/int(participates[3])))
                data.append(int(int(participates[5])*100/int(participates[3])))

            data.extend(dist_rec)
            assert len(data) == 17
            return data
    print("can not find %s in %s" % (name, filename))
    #return [-1] + [-1, -1, -1, -1, -1] + [-1, -1, -1, -1, -1] + [-1, -1, -1, -1, -1, -1]
    return map(lambda x: int(x), [md.hr_days[course]] + md.hr_history_total[course] + md.hr_history_year[course] + md.dist_rec[course])


# 이름  소속 생일        데뷔일  총경기수, 총1, 총2, 1년, 1년1, 1년2
# 김동철491974/11/28371995/07/015252 3706  217  242  166   17   19
def parse_txt_jockey(date, name, course, md=mean_data()):
    course = int(course)
    if len(str(name)) > 9:
        #print("name is changed %s -> %s" % (name, name[:6]))
        name = str(name)[:6]
    filename = get_fname(date, "jockey")
    #print("find %s at %s" % (name, filename))
    f_input = open(filename)
    while True:
        line = f_input.readline()
        line = unicode(line, 'euc-kr').encode('utf-8')
        if len(line) == 0:
            break
        if re.search(unicode(name, 'utf-8').encode('utf-8'), line) is not None:
            data = []
            participates = re.search(unicode(r'(?<=[\d\s]{6})[\s\d,]+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s', 'utf-8').encode('utf-8'), line).group().replace(',', '').split()
            #print(participates)
            if int(participates[0]) == 0:
                data.extend([0] + md.jk_history_total[course][1:])
            else:
                data.append(int(participates[0]))
                data.append(int(participates[1]))
                data.append(int(participates[2]))
                data.append(int(participates[1])*100/int(participates[0]))
                data.append(int(participates[2])*100/int(participates[0]))

            if int(participates[3]) == 0:
                data.extend([0] + md.jk_history_year[course][1:])
            else:
                data.append(int(participates[3]))
                data.append(int(participates[4]))
                data.append(int(participates[5]))
                data.append(int(participates[4])*100/int(participates[3]))
                data.append(int(participates[5])*100/int(participates[3]))

            return data
    print("can not find %s in %s" % (name, filename))
    #return [-1, -1, -1, -1, -1] + [-1, -1, -1, -1, -1]
    return map(lambda x: int(x), md.jk_history_total[course] + md.jk_history_year[course])


# 이름  소속 생일        데뷔일  총경기수, 총1, 총2, 1년, 1년1, 1년2
# 곽영효191961/09/24551997/05/283,868  438  394  134   18   13
def parse_txt_trainer(date, name, course, md=mean_data()):
    if len(str(name)) > 9:
        #print("name is changed %s -> %s" % (name, name[:6]))
        name = str(name)[:6]
    course = int(course)
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
                data.extend([0] + md.tr_history_total[course][1:])
            else:
                data.append(int(participates[0]))
                data.append(int(participates[1]))
                data.append(int(participates[2]))
                data.append(int(participates[1])*100/int(participates[0]))
                data.append(int(participates[2])*100/int(participates[0]))

            if int(participates[3]) == 0:
                data.extend([0] + md.tr_history_year[course][1:])
            else:
                data.append(int(participates[3]))
                data.append(int(participates[4]))
                data.append(int(participates[5]))
                data.append(int(participates[4])*100/int(participates[3]))
                data.append(int(participates[5])*100/int(participates[3]))

            return data
    print("can not find %s in %s" % (name, filename))
    #return [-1, -1, -1, -1, -1] + [-1, -1, -1, -1, -1]
    return map(lambda x: int(x), md.tr_history_total[course] + md.tr_history_year[course])


def get_data(filename, md=mean_data()):
    print("race file: %s" % filename)
    date_i = re.search(unicode(r'\d{8}', 'utf-8').encode('utf-8'), filename).group()
    date = datetime.date(int(date_i[:4]), int(date_i[4:6]), int(date_i[6:]))
    data = parse_txt_race(filename, md)
    for i in range(len(data)):
        #print("race file: %s" % filename)
        #print("%s %s %s" % (data[i][5], data[i][10], data[i][11]))
        data[i].extend(parse_txt_horse(date, int(data[i][39]), data[i][24], data[i][0], md))
        data[i].extend(parse_txt_jockey(date, data[i][29], data[i][0], md))
        data[i].extend(parse_txt_trainer(date, data[i][30], data[i][0], md))
        data[i].extend([date_i])
    df = pd.DataFrame(data)
    df.columns = ['course', 'humidity', 'kind', 'dbudam', 'drweight', 'lastday', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'ts6', # 12
                  'score1', 'score2', 'score3', 'score4', 'score5', 'score6', 'score7', 'score8', 'score9', 'score10', # 10
                  'rank', 'idx', 'name', 'cntry', 'gender', 'age', 'budam', 'jockey', 'trainer', # 9
                  'owner', 'weight', 'dweight', 'rctime', 'r1', 'r2', 'r3', 'cnt', 'rcno', 'month', 'price', 'bokyeon1', 'bokyeon2', 'bokyeon3', 'boksik', 'ssang', 'sambok', 'samssang', # 18
                  'hr_days', 'hr_nt', 'hr_nt1', 'hr_nt2', 'hr_t1', 'hr_t2', 'hr_ny', 'hr_ny1', 'hr_ny2', 'hr_y1', 'hr_y2', # 11
                  'hr_dt', 'hr_d1', 'hr_d2', 'hr_rh', 'hr_rm', 'hr_rl', # 6
                  'jk_nt', 'jk_nt1', 'jk_nt2', 'jk_t1', 'jk_t2', 'jk_ny', 'jk_ny1', 'jk_ny2', 'jk_y1', 'jk_y2', # 10
                  'tr_nt', 'tr_nt1', 'tr_nt2', 'tr_t1', 'tr_t2', 'tr_ny', 'tr_ny1', 'tr_ny2', 'tr_y1', 'tr_y2', 'date'] # 10
    return df


if __name__ == '__main__':
    DEBUG = True
    filename = '../txt/3/rcresult/rcresult_3_20070106.txt'
    md = joblib.load('../data/3_2007_2016_md.pkl')
    data = get_data(filename, md)
    data.to_csv(filename.replace('.txt', '.csv'), index=False)
    del data['name']
    del data['jockey']
    del data['trainer']
    del data['owner']
    del data['rctime']
    del data['rank']
    del data['r3']
    del data['r2']
    del data['r1']
    del data['date']
    del data['price']
    del data['bokyeon1']
    del data['bokyeon2']
    del data['bokyeon3']
    del data['boksik']
    del data['ssang']
    del data['sambok']
    del data['samssang']
    data.to_csv(filename.replace('.txt', '_x.csv'), index=False)

    print(data)




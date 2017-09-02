# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os.path
import glob
from sklearn.externals import joblib

DEBUG = False
def make_one_hot(data):
    jangu_list = ["망사눈가면", "눈가면", "계란형큰고리재갈", "눈귀가면", "망사가면", "반가지큰고리재갈", "혀보정끈", "눈가리개", "양털코굴레", "가지재갈", "구각자극판", "진입보조", "주립보조", "보조", "재갈보정밴드", "반가지재갈", "고정마팅게일변형", "진입밴드", "혀보정재갈", "양털뺨굴레", "양뒷다리편자미착", "견인용재갈", "보조선진입마", "나비재갈", "양앞다리편자미착"]
    clinic_list = ["각막염", "감기", "건단열", "견파행", "계인대염", "계질환", "골막염", "골연골증", "골절", "관골막염", "관절염종장", "관파행", "교돌상", "구절염", "굴건염", "기관지염", "담마진", "마비성근색소뇨", "봉와직염", "비절내종", "비절후종", "산통", "식욕부진", "양견파행", "양전구절염", "열사병", "열제", "열창", "완골", "완관절염", "완관절염", "요배통", "우견파행", "우후답창", "임파관염", "자창", "전지구절염", "전지굴건염", "전지답창", "전지제구염", "전지제염", "전지제차부란", "제구염", "제염", "종자골골절", "좌상", "중수골골절", "찰과상", "천지굴건염", "퇴행성관절염", "파행", "폐출혈", "피로회복", "활막염", "후두염", "후지파행"]
    res = np.zeros(len(jangu_list) + len(clinic_list))
    for i in range(len(jangu_list)):
        for j in data[0]:
            if jangu_list[i] in j:
                res[i] = 1
    for i in range(len(clinic_list)):
        for c in data[1]:
            if clinic_list[i] in c:
                res[len(jangu_list) + i] = 1
    return res

def parse_hr_clinic(date):
    for _ in range(20):
        date += datetime.timedelta(days=-1)
        date_i = int("%d%02d%02d" % (date.year, date.month, date.day))
        filename = "../txt/1/weekly-jangu/weekly-jangu_1_%d.txt" % date_i
        if os.path.isfile(filename):
            break
    if not os.path.isfile(filename):
        return dict()
    if DEBUG: print("open %s" % filename)
    in_data = open(filename)
    data = dict()
    name = ''
    while True:
        line = in_data.readline()
        #line = unicode(line, 'euc-kr').encode('utf-8')
        if line is None or len(line) == 0:
            break

        if re.search(r'보조장구', line) is not None:
            name_list = []
            while re.search(r'[-─]{5}', line) is None:
                line = in_data.readline()
                #line = unicode(line, 'euc-kr').encode('utf-8')
                break
            jangu = []
            clinic = []
            while True:
                line = in_data.readline()
                #line = unicode(line, 'euc-kr').encode('utf-8')
                if DEBUG: print("%s" % unicode(line, 'utf-8'))
                if len(line) == 0 or line == None:
                    if DEBUG: print("break: end of file")
                    break
                if re.search(r'$경주일', line) is not None:
                    if DEBUG: print("break: %s" % line)
                    break
                if re.search(r'\d\s{2}[가-힣]+', line) is not None:
                    if name != '':
                        #data[name] = [jangu, clinic]
                        name = name.replace('★', '')
                        data[name] = make_one_hot([jangu, clinic])
                    name = re.search(r'(?<=\d\s{2})[가-힣]+', line).group()
                    jangu = []
                    clinic = []
                if re.search(r'(?<=\s{3})[가-힣]+(?=\s+\d{4})', line) is not None:
                    jangu.append(re.search(r'(?<=\s{3})[가-힣]+(?=\s+\d{4})', line).group().strip())
                if re.search(r'(?<=\d{4}\.\d{2}\.\d{2})\S+', line) is not None:
                    clinic.extend(re.search(r'(?<=\d{4}\.\d{2}\.\d{2})\S+.+', line).group().strip().split(','))
    data[name] = make_one_hot([jangu, clinic])
    return data  # len: 81

def get_jangu_clinic(data, name):
    try:
        return data[name]
    except KeyError:
        return [0]*81
    except:
        print("Unexpected error in jangu clinic of %s" % unicode(name, 'utf-8'))


if __name__ == '__main__':
    DEBUG = False
    jangus = []
    clinics = []
    for year in range(201601, 201602):
        flist = glob.glob('../txt/1/weekly-jangu/weekly-jangu_1_%d*' % year)
        for fname in flist:
            print("process %s" % fname)
            date_i = int(re.search(r'\d{8}', fname).group())
            data = parse_hr_clinic(datetime.date(date_i/10000, date_i/100%100, date_i%100))
            for k,v in data.iteritems():
                print("%s" % unicode(k, 'utf-8'))
                print(v)
                print(len(v))
    #data = parse_hr_clinic(datetime.date(2010, 4, 22))
    #for k,v in data.iteritems():
    #    print("name: %s" % unicode(k, 'utf-8'))
    #    print("jangu")
    #    for j in v[0]: print(unicode(j, 'utf-8'))
    #    print("clinic")
    #    for c in v[1]: print(unicode(c, 'utf-8'))


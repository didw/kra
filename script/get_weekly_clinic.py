# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os.path
import glob
from sklearn.externals import joblib

DEBUG = False
gJangu = dict()
gClinic = dict()
def make_one_hot(data):
    jangu_list = ["양눈가면", "가지재갈", "망사가면", "망사눈가면", "구각자극판", "양눈귀가면"]
    clinic_list = ["기타임신검사", "감기", "운동기질환", "좌전지파행", "우전지파행", "낭치", "정치", "산통", "식이성 식욕부진", "임신검사", "양전지파행", "외상기타", "좌후지파행", "안상", "우후지파행", "소화기 질환 기타", "우제3중수골골막염", "신경성 식욕부진", "좌전구절염", "좌전굴건염", "의사선역", "양후지파행", "안질환기타", "좌제3중수골골막염", "좌각막염", "술후처치"]
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
        filename = "../txt/2/weekly-jangu/weekly-jangu_2_%d.txt" % date_i
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
        line = unicode(line, 'euc-kr').encode('utf-8')
        if line is None or len(line) == 0:
            break

        if re.search(r'보조장구', line) is not None:
            name_list = []
            while re.search(r'[-─]{5}', line) is None:
                line = in_data.readline()
                line = unicode(line, 'euc-kr').encode('utf-8')
                break
            jangu = []
            clinic = []
            while True:
                line = in_data.readline()
                line = unicode(line, 'euc-kr').encode('utf-8')
                if DEBUG: print("%s" % unicode(line, 'utf-8'))
                if len(line) == 0 or line == None:
                    if DEBUG: print("break: end of file")
                    break
                if re.search(r'$경주일', line) is not None:
                    if DEBUG: print("break: %s" % line)
                    break
                if re.search(r'\d\s{2}[가-힣]+', line) is not None:
                    if name != '':
                        name = name.replace('★', '')
                        #data[name] = [jangu, clinic]
                        data[name] = make_one_hot([jangu, clinic])
                    name = re.search(r'(?<=\d\s{2})[가-힣]+', line).group()
                    jangu = []
                    clinic = []
                if re.search(r'(?<=\s{3})[가-힣]+(?=\s+\d{4})', line) is not None:
                    jangu.append(re.search(r'(?<=\s{3})[가-힣]+(?=\s+\d{4})', line).group().strip())
                    try:
                        gJangu[jangu[-1]] += 1
                    except KeyError:
                        gJangu[jangu[-1]] = 1
                if re.search(r'(?<=\d{4}\.\d{2}\.\d{2})\S+', line) is not None:
                    clinic.extend(re.search(r'(?<=\d{4}\.\d{2}\.\d{2})\S+.+', line).group().strip().split(','))
                    try:
                        gClinic[clinic[-1]] += 1
                    except KeyError:
                        gClinic[clinic[-1]] = 1
    data[name] = make_one_hot([jangu, clinic])
    return data  # len: 32

def get_jangu_clinic(data, name):
    try:
        return data[name]
    except KeyError:
        return [0]*32
    except:
        print("Unexpected error in jangu clinic of %s" % unicode(name, 'utf-8'))


def print_all_data():
    print('===jangu===')
    for k, v in gJangu.iteritems():
        print('%s, %d' % (k, v))
    print('===clinic===')
    for k, v in gClinic.iteritems():
        print('%s, %d' % (k, v))

if __name__ == '__main__':
    DEBUG = False
    jangus = []
    clinics = []
    for year in range(2007, 2017):
        flist = glob.glob('../txt/2/weekly-jangu/weekly-jangu_2_%d*' % year)
        for fname in flist:
            print("process %s" % fname)
            date_i = int(re.search(r'\d{8}', fname).group())
            data = parse_hr_clinic(datetime.date(date_i/10000, date_i/100%100, date_i%100))
    print_all_data()

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
    jangu_list = ["가지재갈", "계란형큰고리재갈", "고정마팅게일변형", "구각자극판", "눈가리개", "눈가면", "눈귀가면", "망사가면", "망사눈가면", "반가지재갈", "반가지큰고리재갈", "보조선진입마", "보조후진입마", "양털뺨굴레", "양털코굴레", "자력선진입마", "재갈보정밴드", "진입밴드", "혀보정끈", "혀보정재갈"] # 20
    clinic_list = ["운동기인성 피로회복", "좌후 구절부 찰과상", "우후 구절부 찰과상", "양제3중수골골막염", "좌제3중수골골막염", "우제3중수골골막염", "일본뇌염예방접종", "인푸렌자예방접종", "운동기질환 기타", "좌중수부 찰과상", "우비절부 찰과상", "식이성 식욕부진", "좌완슬부 찰과상", "거세술 후 처치", "우중족부찰과상", "좌중족부찰과상", "우중수부찰과상", "좌전지 근육통", "우전지 근육통", "양전지 근육통", "좌후계부찰과상", "좌비절부찰과상", "우전구절찰과상", "우후계부찰과상", "호흡기질환기타", "좌전구절찰과상", "우완슬부찰과상", "선역예방접종", "좌전지 부종", "우전지 부종", "좌각막찰과상", "우각막찰과상", "좌후지 부종", "진정(장제)", "우후지 부종", "양후지 부종", "양각막찰과상", "좌전지파행", "우전지파행", "좌후지파행", "우후지파행", "양전지파행", "양전구절염", "좌전구절염", "건강검진", "마체검사", "외상기타", "술후처치", "의사선역", "근육통", "찰과상", "요배통", "담마진", "각막염", "피부염", "교돌상", "거세술", "정치", "발치", "감기", "산통"]
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
        filename = "../txt/3/weekly-jangu/weekly-jangu_3_%d.txt" % date_i
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
                        #data[name] = [jangu, clinic]
                        name = name.replace('★', '')
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
    return data  # len: 81

def get_jangu_clinic(data, name):
    try:
        return data[name]
    except KeyError:
        return [0]*81
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
        flist = glob.glob('../txt/3/weekly-jangu/weekly-jangu_3_%d*' % year)
        for fname in flist:
            print("process %s" % fname)
            date_i = int(re.search(r'\d{8}', fname).group())
            data = parse_hr_clinic(datetime.date(date_i/10000, date_i/100%100, date_i%100))
    print_all_data()

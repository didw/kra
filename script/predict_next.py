#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob
import parse_xml_entry as xe
import datetime
import train as tr
import train_keras as tk
import train_tf_ensenble as tfn
import train_keras_ensenble as tkn
from get_race_detail import RaceDetail
import numpy as np
import os

def normalize_data(org_data, nData=47):
    data = org_data.dropna()
    data = data.reset_index()
    data.loc[data['gender'] == '암', 'gender'] = 0
    data.loc[data['gender'] == '수', 'gender'] = 1
    data.loc[data['gender'] == '거', 'gender'] = 2
    data.loc[data['cntry'] == '한', 'cntry'] = 0
    data.loc[data['cntry'] == '한(포)', 'cntry'] = 1
    data.loc[data['cntry'] == '제', 'cntry'] = 2
    data.loc[data['cntry'] == '일', 'cntry'] = 3
    data.loc[data['cntry'] == '중', 'cntry'] = 4
    data.loc[data['cntry'] == '미', 'cntry'] = 5
    data.loc[data['cntry'] == '캐', 'cntry'] = 6
    data.loc[data['cntry'] == '뉴', 'cntry'] = 7
    data.loc[data['cntry'] == '호', 'cntry'] = 8
    data.loc[data['cntry'] == '브', 'cntry'] = 9
    data.loc[data['cntry'] == '헨', 'cntry'] = 10
    data.loc[data['cntry'] == '남', 'cntry'] = 11
    data.loc[data['cntry'] == '아일', 'cntry'] = 12
    data.loc[data['cntry'] == '모', 'cntry'] = 13
    data.loc[data['cntry'] == '영', 'cntry'] = 14
    data.loc[data['cntry'] == '인', 'cntry'] = 15
    data.loc[data['cntry'] == '아', 'cntry'] = 16
    data.loc[data['cntry'] == '프', 'cntry'] = 17
    data.loc[data['cntry'] == '산', 'cntry'] = 18
    data.loc[data['cntry'] == '래', 'cntry'] = 19

    oh_course = [[0]*10 for _ in range(len(data))]
    oh_gen = [[0]*3 for _ in range(len(data))]
    oh_cnt = [[0]*20 for _ in range(len(data))]
    course_list = [400, 800, 900, 1000, 1110, 1200, 1400, 1610, 1700, 1800]
    for i in range(len(data)):
        oh_course[i][course_list.index(int(data['course'][i]))] = 1
        oh_gen[i][data['gender'][i]] = 1
        oh_cnt[i][data['cntry'][i]] = 1
    df_course = pd.DataFrame(oh_course, columns=['cr%d'%i for i in range(1,11)])
    df_gen = pd.DataFrame(oh_gen, columns=['g1', 'g2', 'g3'])
    df_cnt = pd.DataFrame(oh_cnt, columns=['c%d'%i for i in range(1,21)])
    return pd.concat([data, df_course, df_gen, df_cnt], axis=1)

    if nData == 47:
        data = data.drop(['ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'ts6', 'score1', 'score2', 'score3', 'score4', 'score5', 'score6', 'score7', 'score8', 'score9', 'score10', 'hr_dt', 'hr_d1', 'hr_d2', 'hr_rh', 'hr_rm', 'hr_rl'], axis=1)
        data = data.drop(['rd1', 'rd2', 'rd3', 'rd4', 'rd5', 'rd6', 'rd7', 'rd8', 'rd9', 'rd10', 'rd11', 'rd12', 'rd13', 'rd14', 'rd15', 'rd16', 'rd17', 'rd18', # 18
                  'jc1', 'jc2', 'jc3', 'jc4', 'jc5', 'jc6', 'jc7', 'jc8', 'jc9', 'jc10', 'jc11', 'jc12', 'jc13', 'jc14', 'jc15', 'jc16', 'jc17', 'jc18', 'jc19', 'jc20', 'jc21', 'jc22', 'jc23', 'jc24', 'jc25', 'jc26', 'jc27', 'jc28', 'jc29', 'jc30',
                  'jc31', 'jc32', 'jc33', 'jc34', 'jc35', 'jc36', 'jc37', 'jc38', 'jc39', 'jc40', 'jc41', 'jc42', 'jc43', 'jc44', 'jc45', 'jc46', 'jc47', 'jc48', 'jc49', 'jc50', 'jc51', 'jc52', 'jc53', 'jc54', 'jc55', 'jc56', 'jc57', 'jc58', 'jc59', 'jc60',
                  'jc61', 'jc62', 'jc63', 'jc64', 'jc65', 'jc66', 'jc67', 'jc68', 'jc69', 'jc70', 'jc71', 'jc72', 'jc73', 'jc74', 'jc75', 'jc76', 'jc77', 'jc78', 'jc79', 'jc80', 'jc81'], axis=1)
    return data


"""['course', 'humidity', 'kind', 'rank', 'idx', 'name', 'cntry', 'gender', 'age', 'budam', 'jockey', 'trainer', # 12
  'owner', 'weight', 'dweight', 'rctime', 'r1', 'r2', 'r3', 'cnt', 'rcno', 'month', # 10
  'hr_days', 'hr_nt', 'hr_nt1', 'hr_nt2', 'hr_t1', 'hr_t2', 'hr_ny', 'hr_ny1', 'hr_ny2', 'hr_y1', 'hr_y2', # 11
  'hr_dt', 'hr_d1', 'hr_d2', 'hr_rh', 'hr_rm', 'hr_rl', # 6
  'jk_nt', 'jk_nt1', 'jk_nt2', 'jk_t1', 'jk_t2', 'jk_ny', 'jk_ny1', 'jk_ny2', 'jk_y1', 'jk_y2', # 10
  'tr_nt', 'tr_nt1', 'tr_nt2', 'tr_t1', 'tr_t2', 'tr_ny', 'tr_ny1', 'tr_ny2', 'tr_y1', 'tr_y2'] # 10
  """
def print_log(data, pred, fname):
    flog = open(fname, 'w')
    rcno = 1
    flog.write("rcno\tcourse\tidx\tname\tcntry\tgender\tage\tbudam\tjockey\ttrainer\tweight\tdweight\thr_days\thumidity\thr_nt\thr_nt1\thr_nt2\thr_ny\thr_ny1\thr_ny2\t")
    flog.write("jk_nt\tjk_nt1\tjk_nt2\tjk_ny\tjk_ny1\tjk_ny2\ttr_nt\ttr_nt1\ttr_nt2\ttr_ny\ttr_ny1\ttr_ny2\tpredict\n")
    for idx in range(len(data)):
        if rcno != data['rcno'][idx]:
            rcno = data['rcno'][idx]
            flog.write('\n')
        flog.write("%s\t%s\t%s\t%s\t%s\t" % (data['rcno'][idx], data['course'][idx], data['idx'][idx], data['name'][idx], data['cntry'][idx]))
        flog.write("%s\t%s\t%s\t%s\t%s\t" % (data['gender'][idx], data['age'][idx], data['budam'][idx], data['jockey'][idx], data['trainer'][idx]))
        flog.write("%s\t%s\t%s\t%s\t" % (data['weight'][idx], data['dweight'][idx], data['hr_days'][idx], data['humidity'][idx]))
        flog.write("%s\t%s\t%s\t%s\t%s\t%s\t" % (data['hr_nt'][idx], data['hr_nt1'][idx], data['hr_nt2'][idx], data['hr_ny'][idx], data['hr_ny1'][idx], data['hr_ny2'][idx]))
        flog.write("%s\t%s\t%s\t%s\t%s\t%s\t" % (data['jk_nt'][idx], data['jk_nt1'][idx], data['jk_nt2'][idx], data['jk_ny'][idx], data['jk_ny1'][idx], data['jk_ny2'][idx]))
        flog.write("%s\t%s\t%s\t%s\t%s\t%s\t" % (data['tr_nt'][idx], data['tr_nt1'][idx], data['tr_nt2'][idx], data['tr_ny'][idx], data['tr_ny1'][idx], data['tr_ny2'][idx]))
        flog.write("%f\n" % pred['predict'][idx])
    flog.close()


def get_chulma_fname(date):
    if date.weekday() == 4:
        date_ = date+ datetime.timedelta(days=-2)
    if date.weekday() == 5:
        date_ = date + datetime.timedelta(days=-3)
    return "../txt/2/chulma/chulma_2_%4d%02d%02d.txt" % (date_.year, date_.month, date_.day)


# 300*48 + 2000*6*4 + 14000 + 300*55 = 
def print_detail(players, cand, fresult, mode):
    if cand == [[1,2,3],[1,2,3,4,5],[1,2,3,4,5,6]] and mode == "ss":
        print("bet: 100") # 14200 / 48 = 295
        print("%s,%s,{%s,%s,%s,%s}" % (players[0], players[1], players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[0], players[2], players[1], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[0], players[3], players[1], players[2], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[0], players[4], players[1], players[2], players[3], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[1], players[0], players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[1], players[2], players[0], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[1], players[3], players[0], players[2], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[1], players[4], players[0], players[2], players[3], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[2], players[0], players[1], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[2], players[1], players[0], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[2], players[3], players[0], players[1], players[4], players[5]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[2], players[4], players[0], players[1], players[3], players[5]))
        fresult.write("\n\nbet: 100") # 14200 / 48 = 295
        fresult.write("\n%s,%s,{%s,%s,%s,%s}, %s: 100" % (players[0], players[1], players[2], players[3], players[4], players[5], mode))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}, %s: 100" % (players[0], players[2], players[1], players[3], players[4], players[5], mode))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}, %s: 100" % (players[0], players[3], players[1], players[2], players[4], players[5], mode))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}, %s: 100" % (players[0], players[4], players[1], players[2], players[3], players[5], mode))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}, %s: 100" % (players[1], players[0], players[2], players[3], players[4], players[5], mode))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}, %s: 100" % (players[1], players[2], players[0], players[3], players[4], players[5], mode))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}, %s: 100" % (players[1], players[3], players[0], players[2], players[4], players[5], mode))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}, %s: 100" % (players[1], players[4], players[0], players[2], players[3], players[5], mode))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}, %s: 100" % (players[2], players[0], players[1], players[3], players[4], players[5], mode))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}, %s: 100" % (players[2], players[1], players[0], players[3], players[4], players[5], mode))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}, %s: 100" % (players[2], players[3], players[0], players[1], players[4], players[5], mode))
        fresult.write("\n%s,%s,{%s,%s,%s,%s}, %s: 100" % (players[2], players[4], players[0], players[1], players[3], players[5], mode))
    elif cand == [1,2,3] and mode == "sb":
        print("%s,%s,%s, %s: 1000" % (players[0], players[1], players[2], mode))

        fresult.write("\n%s,%s,%s, %s: 1000" % (players[0], players[1], players[2], mode))
    elif cand == [[1,2],[1,2,3],[1,2,3]] and mode == "ss":
        print("%s,%s,%s, %s: 500" % (players[0], players[1], players[2], mode))
        print("%s,%s,%s, %s: 500" % (players[0], players[2], players[1], mode))
        print("%s,%s,%s, %s: 500" % (players[1], players[0], players[2], mode))
        print("%s,%s,%s, %s: 500" % (players[1], players[2], players[0], mode))

        fresult.write("\n%s,%s,%s, %s: 2600" % (players[0], players[1], players[2], mode))
        fresult.write("\n%s,%s,%s, %s: 2600" % (players[0], players[2], players[1], mode))
        fresult.write("\n%s,%s,%s, %s: 2600" % (players[1], players[0], players[2], mode))
        fresult.write("\n%s,%s,%s, %s: 2600" % (players[1], players[2], players[0], mode))
    elif cand == [[1,2,3],[1,2,3],[1,2,3]] and mode == "ss":
        print("%s,%s,%s, %s: 500" % (players[0], players[1], players[2], mode))
        print("%s,%s,%s, %s: 500" % (players[0], players[2], players[1], mode))
        print("%s,%s,%s, %s: 500" % (players[1], players[0], players[2], mode))
        print("%s,%s,%s, %s: 500" % (players[1], players[2], players[0], mode))
        print("%s,%s,%s, %s: 500" % (players[2], players[0], players[1], mode))
        print("%s,%s,%s, %s: 500" % (players[2], players[1], players[0], mode))

        fresult.write("\n%s,%s,%s, %s: 18000" % (players[0], players[1], players[2], mode))
        fresult.write("\n%s,%s,%s, %s: 3000" % (players[0], players[2], players[1], mode))
        fresult.write("\n%s,%s,%s, %s: 3000" % (players[1], players[0], players[2], mode))
        fresult.write("\n%s,%s,%s, %s: 3000" % (players[1], players[2], players[0], mode))
        fresult.write("\n%s,%s,%s, %s: 3000" % (players[2], players[0], players[1], mode))
        fresult.write("\n%s,%s,%s, %s: 3000" % (players[2], players[1], players[0], mode))
    elif cand == [[4,5,6],[4,5,6],[4,5,6]]:
        print("bet: 2000")  # 14200 / 6 = 2366
        print("%s,%s,%s" % (players[3], players[4], players[5]))
        print("%s,%s,%s" % (players[3], players[5], players[4]))
        print("%s,%s,%s" % (players[4], players[3], players[5]))
        print("%s,%s,%s" % (players[4], players[5], players[3]))
        print("%s,%s,%s" % (players[5], players[3], players[4]))
        print("%s,%s,%s" % (players[5], players[4], players[3]))

        fresult.write("\n\nbet: 700")  # 14200 / 6 = 2366
        fresult.write("\n%s,%s,%s" % (players[3], players[4], players[5]))
        fresult.write("\n%s,%s,%s" % (players[3], players[5], players[4]))
        fresult.write("\n%s,%s,%s" % (players[4], players[3], players[5]))
        fresult.write("\n%s,%s,%s" % (players[4], players[5], players[3]))
        fresult.write("\n%s,%s,%s" % (players[5], players[3], players[4]))
        fresult.write("\n%s,%s,%s" % (players[5], players[4], players[3]))
    elif cand == [[4,5,6],[4,5,6],[4,5,6,7]]:
        print("bet: 1600")  # 14200 / 6 = 2366
        print("%s,%s,%s" % (players[3], players[4], players[5]))
        print("%s,%s,%s" % (players[3], players[5], players[4]))
        print("%s,%s,%s" % (players[4], players[3], players[5]))
        print("%s,%s,%s" % (players[4], players[5], players[3]))
        print("%s,%s,%s" % (players[5], players[3], players[4]))
        print("%s,%s,%s" % (players[5], players[4], players[3]))

        fresult.write("\n\nbet: 800")  # 14200 / 6 = 2366
        fresult.write("\n%s,%s,%s, %s: 800" % (players[3], players[4], players[5], mode))
        fresult.write("\n%s,%s,%s, %s: 800" % (players[3], players[4], players[6], mode))
        fresult.write("\n%s,%s,%s, %s: 800" % (players[3], players[5], players[4], mode))
        fresult.write("\n%s,%s,%s, %s: 800" % (players[3], players[5], players[6], mode))
        fresult.write("\n%s,%s,%s, %s: 800" % (players[4], players[3], players[5], mode))
        fresult.write("\n%s,%s,%s, %s: 800" % (players[4], players[3], players[6], mode))
        fresult.write("\n%s,%s,%s, %s: 800" % (players[4], players[5], players[3], mode))
        fresult.write("\n%s,%s,%s, %s: 800" % (players[4], players[5], players[6], mode))
        fresult.write("\n%s,%s,%s, %s: 800" % (players[5], players[3], players[4], mode))
        fresult.write("\n%s,%s,%s, %s: 800" % (players[5], players[3], players[6], mode))
        fresult.write("\n%s,%s,%s, %s: 800" % (players[5], players[4], players[3], mode))
        fresult.write("\n%s,%s,%s, %s: 800" % (players[5], players[4], players[6], mode))
    elif cand == [[4,5,6,7,8],[4,5,6,7,8],[4,5,6,7,8]] and mode == "ss":
        print("bet: 100")  # 14200 / 60 = 200
        print("%s,%s,{%s,%s,%s}" % (players[3], players[4], players[5], players[6], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[3], players[5], players[4], players[6], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[3], players[6], players[4], players[5], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[3], players[7], players[4], players[5], players[6]))
        print("%s,%s,{%s,%s,%s}" % (players[4], players[3], players[5], players[6], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[4], players[5], players[3], players[6], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[4], players[6], players[3], players[5], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[4], players[7], players[3], players[5], players[6]))
        print("%s,%s,{%s,%s,%s}" % (players[5], players[3], players[4], players[6], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[5], players[4], players[3], players[6], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[5], players[6], players[3], players[4], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[5], players[7], players[3], players[4], players[6]))
        print("%s,%s,{%s,%s,%s}" % (players[6], players[3], players[4], players[5], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[6], players[4], players[3], players[5], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[6], players[5], players[3], players[4], players[7]))
        print("%s,%s,{%s,%s,%s}" % (players[6], players[7], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[7], players[3], players[4], players[5], players[6]))
        print("%s,%s,{%s,%s,%s}" % (players[7], players[4], players[3], players[5], players[6]))
        print("%s,%s,{%s,%s,%s}" % (players[7], players[5], players[3], players[4], players[6]))
        print("%s,%s,{%s,%s,%s}" % (players[7], players[6], players[3], players[4], players[5]))

        fresult.write("\n\nbet: 100")  # 14200 / 60 = 200
        for x in [4,5,6,7,8]:
            for y in [4,5,6,7,8]:
                for z in [4,5,6,7,8]:
                    if x == y or x == z or y ==z: continue
                    fresult.write("\n%d,%d,%d, %s: 100" % (x,y,z, mode))
    elif cand == [[1],[2],[3]]:
        print("bet: 4000")  # 14200
        print("%s,%s,%s" % (players[0], players[1], players[2]))
        fresult.write("\n\nbet: 4000")  # 14200
        fresult.write("\n%s,%s,%s, %s: 20000" % (players[0], players[1], players[2], mode))
    elif cand == [[1,2,3,4],[1,2,3,4,5,6],[3,4,5,6]]:
        print("bet: 100") # 14200 / 55 = 258
        print("%s,%s,{%s,%s,%s,%s}" % (players[0], players[1], players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[0], players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[0], players[3], players[2], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[0], players[4], players[2], players[3], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[0], players[5], players[2], players[3], players[4]))
        print("%s,%s,{%s,%s,%s,%s}" % (players[1], players[0], players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[1], players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[1], players[3], players[2], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[1], players[4], players[2], players[3], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[1], players[5], players[2], players[3], players[4]))
        print("%s,%s,{%s,%s,%s}" % (players[2], players[0], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s,%s}" % (players[2], players[1], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s}" % (players[2], players[3], players[4], players[5]))
        print("%s,%s,{%s,%s}" % (players[2], players[4], players[3], players[5]))
        print("%s,%s,{%s,%s}" % (players[2], players[5], players[3], players[4]))
        print("%s,%s,{%s,%s,%s}" % (players[3], players[0], players[2], players[4], players[5]))
        print("%s,%s,{%s,%s}" % (players[3], players[1], players[4], players[5]))
        print("%s,%s,{%s,%s}" % (players[3], players[2], players[4], players[5]))
        print("%s,%s,{%s,%s}" % (players[3], players[4], players[2], players[5]))
        print("%s,%s,{%s,%s}" % (players[3], players[5], players[2], players[4]))

        fresult.write("\n\nbet: 100")  # 14200 / 55 = 258
        for x in [1,2,3,4]:
            for y in [1,2,3,4,5,6]:
                for z in [3,4,5,6]:
                    if x == y or x == z or y ==z: continue
                    fresult.write("\n%s,%s,%s, %s: 50" % (players[x-1],players[y-1],players[z-1], mode))


def print_bet(rcdata, course=0, year=4, nData=47, train_course=0):
    print("dan")
    print("%s" % (rcdata['idx'][0]))
    print("bok")
    print("%s,%s" % (rcdata['idx'][0], rcdata['idx'][1]))
    print("bokyeon")
    print("%s,%s" % (rcdata['idx'][0], rcdata['idx'][1]))
    print("ssang")
    print("%s,{%s,%s}" % (rcdata['idx'][0], rcdata['idx'][1], rcdata['idx'][2]))
    print("%s,{%s,%s}" % (rcdata['idx'][1], rcdata['idx'][0], rcdata['idx'][2]))

    print("sambok")
    print("%s,%s,{%s,%s}" % (rcdata['idx'][0], rcdata['idx'][1], rcdata['idx'][2], rcdata['idx'][3]))
    print("{%s,%s},%s,%s" % (rcdata['idx'][0], rcdata['idx'][1], rcdata['idx'][2], rcdata['idx'][3]))

    print("samssang")
    global fname
    fresult = open(fname, 'a')
    fresult.write("%s,%s,%s,%s,%s,%s\n" % (rcdata['idx'][0], rcdata['idx'][1], rcdata['idx'][2], rcdata['idx'][3], rcdata['idx'][4], rcdata['idx'][5]))
    print_detail(rcdata['idx'], [[1,2],[1,2,3],[1,2,3]], fresult, "ss")
    print_detail(rcdata['idx'], [[4,5,6],[4,5,6],[4,5,6,7]], fresult, "ss")
    fresult.close()


def predict_next(estimator, md, rd, meet, date, rcno, course=0, nData=47, year=4, train_course=0):
    data_pre = xe.parse_xml_entry(meet, date, rcno, md, rd)
    data = normalize_data(data_pre, nData=nData)
    print(len(data.columns))
    X_data = data.copy()
    print(len(X_data.columns))
    del X_data['name']
    del X_data['jockey']
    del X_data['trainer']
    del X_data['owner']
    del X_data['index']
    __DEBUG__ = True
    if __DEBUG__:
        X_data.to_csv('../log/predict_x_%d_m%d_r%d.csv' % (date, meet, rcno), index=False)
    print(len(X_data.columns))
    X_array = np.array(X_data)
    pred = pd.DataFrame(estimator.predict(X_array))
    pred.columns = ['predict']
    __DEBUG__ = True
    if __DEBUG__:
        pd.concat([data_pre, pred], axis=1).to_csv('../log/predict_%d_m%d_r%d.csv' % (date, meet, rcno), index=False)
        X_data.to_csv('../log/predict_x_%d_m%d_r%d.csv' % (date, meet, rcno), index=False)
    prev_rc = data['rcno'][0]
    rcdata = []
    for idx, row in data.iterrows():
        if int(data['hr_nt'][idx]) == 0 or int(data['jk_nt'][idx]) == 0 or int(data['tr_nt'][idx]) == 0:
            print("%s data is not enough. be careful[hr:%d, jk:%d, tr:%d]" % (data['name'][idx], int(data['hr_nt'][idx]), int(data['jk_nt'][idx]), int(data['tr_nt'][idx])))
        if row['rcno'] != prev_rc or idx+1 == len(data):
            if idx+1 == len(data):
                rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])
            rcdata = pd.DataFrame(rcdata)
            rcdata.columns = ['idx', 'name', 'time']
            rcdata = rcdata.sort_values(by='time')
            rcdata = rcdata.reset_index(drop=True)
            print("=========== %s ==========" % prev_rc)
            print(rcdata)
            fresult = open(fname, 'a')
            fresult.write("\n\n\n=== rcno: %d, nData: %d, year: %d, train_course: %d ===\n" % (int(prev_rc), nData, year, train_course))
            fresult.close()
            print_bet(rcdata, course, nData=nData, year=year, train_course=train_course)
            rcdata = []
            prev_rc = row['rcno']
            if idx+1 != len(data):
                rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])
        else:
            rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])
    #print(X_data.columns)
    #print(estimator.feature_importances_)

def predict_next_ens(estimators_, md, rd, meet, date, rcno, course=0, nData=47, year=4, train_course=0, scaler=None):
    data_pre = xe.parse_xml_entry(meet, date, rcno, md, rd)
    data = normalize_data(data_pre, nData=nData)
    print(len(data.columns))
    X_data = data.copy()
    print(len(X_data.columns))
    del X_data['name']
    del X_data['jockey']
    del X_data['trainer']
    del X_data['owner']
    del X_data['index']
    __DEBUG__ = True
    if __DEBUG__:
        X_data.to_csv('../log/predict_x_%d_m%d_r%d.csv' % (date, meet, rcno), index=False)
    print(len(X_data.columns))
    X_array = np.array(X_data)
    X_array = scaler[0].transform(X_array)

    for e in range(5):
        estimators = estimators_[e*6:(e+1)*6]
        preds = [0]*len(estimators)
        for i in range(len(estimators)):
            preds[i] = estimators[i].predict(X_array)

        for i in range(len(preds)+1):
            if i == len(preds):
                pred = np.mean(preds, axis=0)
            else:
                pred = preds[i]
                pred = scaler[1].inverse_transform(pred)
                continue
            pred = pd.DataFrame(pred)
            pred.columns = ['predict']
            __DEBUG__ = True
            if __DEBUG__:
                pd.concat([data_pre, pred], axis=1).to_csv('../log/predict_%d_m%d_r%d_%d.csv' % (date, meet, rcno, i), index=False)
                X_data.to_csv('../log/predict_x_%d_m%d_r%d_%d.csv' % (date, meet, rcno, i), index=False)
            prev_rc = data['rcno'][0]
            rcdata = []
            for idx, row in data.iterrows():
                if int(data['hr_nt'][idx]) == 0 or int(data['jk_nt'][idx]) == 0 or int(data['tr_nt'][idx]) == 0:
                    print("%s data is not enough. be careful[hr:%d, jk:%d, tr:%d]" % (
                        data['name'][idx], int(data['hr_nt'][idx]), int(data['jk_nt'][idx]), int(data['tr_nt'][idx])))
                if row['rcno'] != prev_rc or idx+1 == len(data):
                    if idx+1 == len(data):
                        rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])
                    rcdata = pd.DataFrame(rcdata)
                    rcdata.columns = ['idx', 'name', 'time']
                    rcdata = rcdata.sort_values(by='time')
                    rcdata = rcdata.reset_index(drop=True)
                    print("=========== %s ==========" % prev_rc)
                    print(rcdata)
                    fresult = open(fname, 'a')
                    fresult.write("\n\n\n=== rcno: %d, nData: %d, year: %d, train_course: %d, model: %d ===\n" % (int(prev_rc), nData, year, train_course, i))
                    fresult.close()
                    print_bet(rcdata, course, nData=nData, year=year, train_course=train_course)
                    rcdata = []
                    prev_rc = row['rcno']
                    if idx+1 != len(data):
                        rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])
                else:
                    rcdata.append([row['idx'], row['name'], float(pred['predict'][idx])])


def get_race_detail(date):
    rd = RaceDetail()
    import glob
    for year in range(date/10000 - 3, date/10000+1):
        filelist2 = glob.glob('../txt/2/rcresult/rcresult_2_%d*.txt' % year)
        print("loading rslt at %d" % year)
        print("loading rcresult at %d" % year)
        for fname in filelist2:
            rd.parse_race_detail(fname)
    return rd

if __name__ == '__main__':
    meet = 2
    date = 20170407
    train_course = 0
    courses = [0,0,0,0,0,0,0,0,0,0,0,0]
    rcno = 0
    #for rcno in range(len(courses)):
    course = courses[rcno]
    test_course = course
    rd = get_race_detail(date)
    for nData, year, train_course in zip([160], [8], [0]):
        if train_course == 1: train_course = course
        print("Process in train: %d, ndata: %d, year: %d" % (train_course, nData, year))
        #estimator, md = tk.training(datetime.date(date/10000, date/100%100, date%100) + datetime.timedelta(days=-365*year), datetime.date(date/10000, date/100%100, date%100) + datetime.timedelta(days=-1), train_course, nData)
        #predict_next(estimator, md, rd, meet, date, rcno, test_course, nData, year, train_course)

        estimators, md, scaler = tkn.training(datetime.date(date/10000, date/100%100, date%100) + datetime.timedelta(days=-365*year-1), datetime.date(date/10000, date/100%100, date%100) + datetime.timedelta(days=-1), train_course, nData)
        fname = '../result/1704/%d_1.txt' % (date%100,)
        os.system("rm %s" % fname)
        predict_next_ens(estimators, md, rd, meet, date, rcno, test_course, nData, year, train_course, scaler)
        date += 1
        fname = '../result/1704/%d_1.txt' % (date%100,)
        os.system("rm %s" % fname)
        predict_next_ens(estimators, md, rd, meet, date, rcno, test_course, nData, year, train_course, scaler)

# Strategy
# v2 y8 1,2,3

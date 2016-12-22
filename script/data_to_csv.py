#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import parse_txt_race as pr
import datetime
import pandas as pd
import os.path
from mean_data import mean_data
from sklearn.externals import joblib

def get_data(begin_date, end_date, fname_csv):
    train_bd = begin_date
    train_ed = end_date
    date = train_bd
    data = pd.DataFrame()
    first = True
    date += datetime.timedelta(days=-1)
    md = mean_data()
    while date < train_ed:
        date += datetime.timedelta(days=1)
        if date.weekday() != 4 and date.weekday() != 6:
            continue
        for i in [0]:
            print("%f" % md.race_score[i][20], end=' ')
        print()
        #print("value: %f, %f, %f, %f, %f" % (md.race_score[1000][20], md.race_score[1200][20], md.race_score[1300][20], md.race_score[1400][20], md.race_score[1700][20]))
        filename = "../txt/1/rcresult/rcresult_1_%02d%02d%02d.txt" % (date.year, date.month, date.day)
        if not os.path.isfile(filename):
            continue
        if first:
            adata = pr.get_data(filename, md)
            md.update_data(adata)
            data = adata
            first = False
        else:
            adata = pr.get_data(filename, md)
            md.update_data(adata)
            data = data.append(adata, ignore_index=True)
    data.to_csv(fname_csv, index=False)
    joblib.dump(md, fname_csv.replace('.csv', '_md.pkl'))
    return data


def update_data(end_date, fname_csv):
    data = pd.read_csv(fname_csv)
    print("update data from %d to today" % data.loc[len(data)-1]['date'])
    train_bd = data.loc[len(data)-1]['date']
    train_ed = end_date
    date = datetime.date(train_bd/10000, train_bd/100%100, train_bd%100)
    fname_md = fname_csv.replace('.csv', '_md.pkl')
    md = joblib.load(fname_md)
    while date < train_ed:
        date += datetime.timedelta(days=1)
        if date.weekday() != 4 and date.weekday() != 6:
            continue
        for i in [0]:
            print("%f" % md.race_score[i][20], end=' ')
        print()
        filename = "../txt/1/rcresult/rcresult_1_%02d%02d%02d.txt" % (date.year, date.month, date.day)
        if not os.path.isfile(filename):
            continue
        adata = pr.get_data(filename, md)
        md.update_data(adata)
        data = data.append(adata, ignore_index=True)
    os.system("rename \"%s\" \"%s\"" % (fname_csv, fname_csv.replace('.csv', '_%s.csv'%end_date)))
    os.system("rename \"%s\" \"%s\"" % (fname_md, fname_md.replace('.pkl', '_%s.pkl'%end_date)))
    data.to_csv(fname_csv, index=False)
    joblib.dump(md, fname_md)
    return data


def update_md(fname):
    data = pd.read_csv(fname)
    md = mean_data()
    md.update_data(data)
    joblib.dump(md, fname.replace('.csv', '_md.pkl'))


if __name__ == '__main__':
    DEBUG = True
    fname_csv = '../data/3_2007_2016.csv'
    bdate = datetime.date(2007, 1, 1)
    edate = datetime.date(2016, 12, 1)
    get_data(bdate, edate, fname_csv)
    #update_data(datetime.date.today(), fname_csv)


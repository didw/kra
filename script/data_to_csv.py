#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import parse_txt_race as pr
import datetime
import pandas as pd
import os.path
from mean_data import mean_data
from sklearn.externals import joblib
from mean_data3 import save_mean_data
from race_record import RaceRecord
import gzip
import cPickle
import get_lineage as lg
from make_unique_columns import make_unique_columns


def get_data(begin_date, end_date, fname_csv):
    train_bd = begin_date
    train_ed = end_date
    date = train_bd
    data = pd.DataFrame()
    first = True
    date += datetime.timedelta(days=-1)
    fname_md = fname_csv.replace('.csv', '_md.pkl')
    if os.path.isfile(fname_md):
        md = joblib.load(fname_md)
    else:
        md = mean_data()
    with gzip.open('../data/2_2007_2016_v1_md3.gz', 'rb') as f:
        md3 = cPickle.loads(f.read())
    md3['humidity'][20] = md3['humidity'][25]
    while date < train_ed:
        date += datetime.timedelta(days=1)
        if date.weekday() != 4 and date.weekday() != 5:
            continue
        filename = "../txt/2/rcresult/rcresult_2_%02d%02d%02d.txt" % (date.year, date.month, date.day)
        if not os.path.isfile(filename):
            continue
        for i in [300, 800, 900, 1000, 1200, 1400, 0]:
            print("%f, " % md.race_score[i][0][20], end=' ')
        print()
        for i in [300, 800, 900, 1000, 1200, 1400]:
            print("[%.0f %.0f %.0f]" % (md.race_detail[i][0], md.race_detail[i][1], md.race_detail[i][2]), end=', ')
        print()
        if first:
            adata = pr.get_data(filename, md, md3)
            md.update_data(adata)
            data = adata
            first = False
        else:
            adata = pr.get_data(filename, md, md3)
            md.update_data(adata)
            data = data.append(adata, ignore_index=True)
    data.to_csv(fname_csv, index=False)
    return data


def update_data(end_date, fname_csv):
    data = pd.read_csv(fname_csv)
    print("update data from %d to today" % data.loc[len(data)-1]['date'])
    train_bd = data.loc[len(data)-1]['date']
    train_ed = end_date
    date = datetime.date(train_bd/10000, train_bd/100%100, train_bd%100)
    fname_md = fname_csv.replace('.csv', '_md.pkl')
    md = joblib.load(fname_md)

    with gzip.open('../data/1_2007_2016_v1_md3.gz', 'rb') as f:
        md3 = cPickle.loads(f.read())
    md3['humidity'][20] = md3['humidity'][25]

    while date <= train_ed:
        date += datetime.timedelta(days=1)
        if date.weekday() != 4 and date.weekday() != 5:
            continue
        filename = "../txt/2/rcresult/rcresult_2_%02d%02d%02d.txt" % (date.year, date.month, date.day)
        if not os.path.isfile(filename):
            continue
        for i in [300, 800, 900, 1000, 1200, 1400, 0]:
            print("%f" % md.race_score[i][0][20], end=' ')
        print()
        adata = pr.get_data(filename, md, md3)
        data = data.append(adata, ignore_index=True)
    os.system("mv \"%s\" \"%s\"" % (fname_csv, fname_csv.replace('.csv', '_%s.csv'%train_bd)))
    data.to_csv(fname_csv, index=False)
    return data


if __name__ == '__main__':
    DEBUG = True
    fname_csv = '../data/2_2007_2016_v1.csv'
    bdate = datetime.date(2007, 1, 1)
    edate = datetime.date(2016,12,31)
    #get_data(bdate, edate, fname_csv)
    update_data(datetime.date.today(), fname_csv)

    race_record = RaceRecord()
    race_record.load_model()
    race_record.update_model()

    save_mean_data()

    lg.save_lineage_info(1)

    make_unique_columns()

    print("\n\nAll Finished")
    
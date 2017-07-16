#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import parse_txt_race as pr
import datetime
import pandas as pd
import os.path
from mean_data import mean_data
from mean_data3 import cmake_mean
from sklearn.externals import joblib
from get_race_detail import RaceDetail
import multiprocessing as mp
import Queue
import numpy as np
import gzip, cPickle

PROCESS_NUM = 8

def load_worker(worker_idx, filename_queue, output_queue, md, md3):
    print("[W%d] Current File/Feature Queue Size = %d/%d" % (worker_idx, filename_queue.qsize(), output_queue.qsize()))
    afile = filename_queue.get(True, 10)
    adata = pr.get_data(afile, md, md3)
    output_queue.put(adata, True)


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
    with gzip.open('../data/1_2007_2016_v1_md3.gz', 'rb') as f:
        md3 = cPickle.loads(f.read())
    md3['humidity'][20] = md3['humidity'][25]

    filename_queue = mp.Queue()
    data_queue = mp.Queue()

    while date < train_ed:
        date += datetime.timedelta(days=1)
        if date.weekday() != 5 and date.weekday() != 6:
            continue
        filename = "../txt/1/rcresult/rcresult_1_%02d%02d%02d.txt" % (date.year, date.month, date.day)
        if not os.path.isfile(filename):
            continue
        filename_queue.put(filename)

    worker_num = filename_queue.qsize()

    iter_save = 0
    while True:
        print("current index: %d" % worker_num)
        if worker_num < filename_queue.qsize() + PROCESS_NUM and filename_queue.qsize() > 0:
            proc = mp.Process(target=load_worker, args=(worker_num, filename_queue, data_queue, md, md3))
            proc.start()
        try:
            adata = data_queue.get(True, 10)
            if first:
                data = adata
                first = False
            else:
                data = data.append(adata, ignore_index=True)
                iter_save += 1
                if iter_save % 10 == 0:
                    data.to_csv(fname_csv, index=False)
            worker_num -= 1
            print("data get: %d" % worker_num)
        except Queue.Empty:
            print("queue empty.. nothing to get data %d" % filename_queue.qsize())
        if worker_num == 0:
            print("feature extraction finished")
            break
    data.to_csv(fname_csv, index=False)
    return data



if __name__ == '__main__':
    DEBUG = True
    fname_csv = '../data/1_2007_2016_v1.csv'
    bdate = datetime.date(2006, 1, 1)
    edate = datetime.date(2017, 5, 30)
    get_data(bdate, edate, fname_csv)
    #update_data(datetime.date.today(), fname_csv)


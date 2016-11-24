#!/usr/bin/python
# -*- coding: utf-8 -*-

import parse_txt_race as pr
import datetime
import pandas as pd
import os.path


def get_data(begin_date, end_date, fname_csv):
    train_bd = begin_date
    train_ed = end_date
    date = train_bd
    data = pd.DataFrame()
    first = True
    date += datetime.timedelta(days=-1)
    while date < train_ed:
        date += datetime.timedelta(days=1)
        if date.weekday() != 5 and date.weekday() != 6:
            continue
        filename = "../txt/1/rcresult/rcresult_1_%02d%02d%02d.txt" % (date.year, date.month, date.day)
        if not os.path.isfile(filename):
            continue
        if first:
            data = pr.get_data(filename)
            first = False
        else:
            data = data.append(pr.get_data(filename), ignore_index=True)
    data.to_csv(fname_csv, index=False)
    return data


def update_data(end_date, fname_csv):
    data = pd.read_csv(fname_csv)
    print("update data from %d to today" % data.loc[len(data)-1]['date'])
    train_bd = data.loc[len(data)-1]['date']
    train_ed = end_date
    date = datetime.date(train_bd/10000, train_bd/100%100, train_bd%100)
    while date < train_ed:
        date += datetime.timedelta(days=1)
        if date.weekday() != 5 and date.weekday() != 6:
            continue
        filename = "../txt/1/rcresult/rcresult_1_%02d%02d%02d.txt" % (date.year, date.month, date.day)
        if not os.path.isfile(filename):
            continue
        data = data.append(pr.get_data(filename), ignore_index=True)
    data.to_csv(fname_csv.replace('.csv', '_new.csv'), index=False)
    return data


if __name__ == '__main__':
    begin_date = datetime.date(2007, 1, 1)
    end_date = datetime.date.today()
    fname_csv = '../data/1_2007_2016_v1.2.csv'
    get_data(begin_date, end_date, fname_csv)
    #update_data(end_date, fname_csv)

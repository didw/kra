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
            data = pr.get_data_w_date(filename)
            first = False
        else:
            data = data.append(pr.get_data_w_date(filename), ignore_index=True)
    data.to_csv(fname_csv, index=False)
    return data


if __name__ == '__main__':
    import get_txt
    get_txt.download_txt(datetime.date.today() + datetime.timedelta(days=-1), datetime.date.today(), 1, True)
    begin_date = datetime.date.today() + datetime.timedelta(days=-400)
    end_date = datetime.date.today()
    fname_csv = '../data/1_recent_1year.csv'
    get_data(begin_date, end_date, fname_csv)
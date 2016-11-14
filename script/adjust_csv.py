# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import os.path

data = pd.read_csv('../data/1_2007_2016.1.0.csv')
newdata = []
hr_data = {}
jk_data = {}
tr_data = {}


for idx, row in data.iterrows():
    if row['hr_nt'] == -1 and row['name'] in hr_data:
        print(hr_data[row['name']])
        row['hr_nt'] = hr_data[row['name']][0]
        row['hr_nt1'] = hr_data[row['name']][1]
        row['hr_nt2'] = hr_data[row['name']][2]
        row['hr_t1'] = hr_data[row['name']][3]
        row['hr_t2'] = hr_data[row['name']][4]
        row['hr_ny'] = hr_data[row['name']][5]
        row['hr_ny1'] = hr_data[row['name']][6]
        row['hr_ny2'] = hr_data[row['name']][7]
        row['hr_y1'] = hr_data[row['name']][8]
        row['hr_y2'] = hr_data[row['name']][9]
    hr_data[row['name']] = [row['hr_nt'], row['hr_nt1'], row['hr_nt2'], row['hr_t1'], row['hr_t2'], row['hr_ny'], row['hr_ny1'], row['hr_ny2'], row['hr_y1'], row['hr_y2']]
    if row['jk_nt'] == -1 and row['jockey'] in jk_data:
        print(jk_data[row['jockey']])
        row['jk_nt'] = jk_data[row['jockey']][0]
        row['jk_nt1'] = jk_data[row['jockey']][1]
        row['jk_nt2'] = jk_data[row['jockey']][2]
        row['jk_t1'] = jk_data[row['jockey']][3]
        row['jk_t2'] = jk_data[row['jockey']][4]
        row['jk_ny'] = jk_data[row['jockey']][5]
        row['jk_ny1'] = jk_data[row['jockey']][6]
        row['jk_ny2'] = jk_data[row['jockey']][7]
        row['jk_y1'] = jk_data[row['jockey']][8]
        row['jk_y2'] = jk_data[row['jockey']][9]
    jk_data[row['name']] = [row['jk_nt'], row['jk_nt1'], row['jk_nt2'], row['jk_t1'], row['jk_t2'], row['jk_ny'], row['jk_ny1'], row['jk_ny2'], row['jk_y1'], row['jk_y2']]
    if row['tr_nt'] == -1 and row['trainer'] in tr_data:
        print(tr_data[row['trainer']])
        row['tr_nt'] = tr_data[row['trainer']][0]
        row['tr_nt1'] = tr_data[row['trainer']][1]
        row['tr_nt2'] = tr_data[row['trainer']][2]
        row['tr_t1'] = tr_data[row['trainer']][3]
        row['tr_t2'] = tr_data[row['trainer']][4]
        row['tr_ny'] = tr_data[row['trainer']][5]
        row['tr_ny1'] = tr_data[row['trainer']][6]
        row['tr_ny2'] = tr_data[row['trainer']][7]
        row['tr_y1'] = tr_data[row['trainer']][8]
        row['tr_y2'] = tr_data[row['trainer']][9]
    tr_data[row['name']] = [row['tr_nt'], row['tr_nt1'], row['tr_nt2'], row['tr_t1'], row['tr_t2'], row['tr_ny'], row['tr_ny1'], row['tr_ny2'], row['tr_y1'], row['tr_y2']]
data.to_csv('../data/1_2007_2016.1.1.csv', index=False)

#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob
import parse_xml_entry as xe
import parse_xml_jk as xj
import parse_xml_race as xr
import parse_xml_tr as xt

def predict_next():
    meet = 1
    date = "201610"
    data = xe.parse_xml_entry(meet, date)
    print(data)
    df = pd.DataFrame()
    df.columns = ['course', 'humidity', 'rank', 'idx', 'name', 'cntry', 'gender', 'age', 'budam', 'jockey', \
                  'trainer', 'owner', 'weight', 'dweight', 'rctime', 'r1', 'r2', 'r3', 'hr_days', 'hr_t1', \
                  'hr_t2', 'hr_y1', 'hr_y2', 'jk_t1', 'jk_t2', 'jk_y1', 'jk_y2', 'tr_t1', 'tr_t2', 'tr_y1', \
                  'tr_y2']


if __name__ == '__main__':
    meet = 1
    date = 201610
    data = xe.parse_xml_entry(meet, date)
    print data



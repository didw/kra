#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob
import datetime
import os
import re

def parse_txt_hr(meet, date_i):
    data = []
    date = datetime.date(date_i/10000, date_i/100%100, date_i%100)
    for _ in range(30):
        date_i = int("%d%02d%02d" % (date.year, date.month, date.day))
        filename  = '../txt/2/horse/horse_%d_%d.txt' % (meet, date_i)
        date = date + datetime.timedelta(days=-1)
        if os.path.isfile(filename):
            break
    try:
        file_input = open(filename)
    except:
        print("horse file (%d, %d) is not exist" % (meet, date))
        return 0
    print "process in %s" % filename
    for line in file_input.readlines():
        line = unicode(line, 'euc-kr').encode('utf-8')
        p = re.search(unicode(r'\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+', 'utf-8').encode('utf-8'), line).group().replace(',', '').split()
        try:
            gender = re.search(r'(?<=\s)[가-힣]+(?=\d{4})', line).group()
        except:
            print("can not parsing.. %s" % line)
        #print("gender: %s" % (unicode(gender[3:], 'utf-8')))
        name = re.search(r'[가-힣]+', line).group()
        birth = re.search(r'\d{4}/\d{2}/\d{2}', line).group()
        hr_days = (date - datetime.date(int(birth[0:4]), int(birth[5:7]), int(birth[8:10]))).days
        cntt, ord1t, ord2t, cnty, ord1y, ord2y = p[0], p[1], p[2], p[3], p[4], p[5]
        data.append([name, gender[3:], hr_days, cntt, ord1t, ord2t, cnty, ord1y, ord2y])

    df = pd.DataFrame(data)
    df.columns = ["hrName", "gender", "hr_days", "cntT", "ord1T", "ord2T", "cntY", "ord1Y", "ord2Y"]
    return df

if __name__ == '__main__':
    meet = 2
    data = parse_txt_hr(meet)
    print(data)

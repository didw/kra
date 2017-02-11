#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import datetime
import re


def parse_txt_tr(meet, date_i):
    data = []
    date = datetime.date(date_i/10000, date_i/100%100, date_i%100)
    for _ in range(30):
        date_i = int("%d%02d%02d" % (date.year, date.month, date.day))
        filename  = '../txt/2/trainer/trainer_%d_%d.txt' % (meet, date_i)
        date = date + datetime.timedelta(days=-1)
        if os.path.isfile(filename):
            break
    try:
        file_input = open(filename)
    except:
        print("trainer file (%d, %d) is not exist" % (meet, date))
        return 0
    print "process in %s" % filename
    for line in file_input.readlines():
        line = unicode(line, 'euc-kr').encode('utf-8')
        p = re.search(unicode(r'(?<=/\d\d)[\d,]+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+', 'utf-8').encode('utf-8'),
                                     line).group().replace(',', '').split()
        name = re.search(r'[가-힣]+', line).group()
        cntt, ord1t, ord2t, cnty, ord1y, ord2y = p[0], p[1], p[2], p[3], p[4], p[5]
        data.append([name, cntt, ord1t, ord2t, cnty, ord1y, ord2y])

    df = pd.DataFrame(data)
    df.columns = ["trName", "cntT", "ord1T", "ord2T", "cntY", "ord1Y", "ord2Y"]
    return df


if __name__ == 'main':
    meet = 1
    data = parse_txt_tr(meet)
    print data

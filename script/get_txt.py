#!/usr/bin/python
# -*- coding: utf-8 -*-

from urllib2 import Request, urlopen
import datetime
import time

# 경마성적표. 토일 업데이트
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/rcresult/20161023dacom11.rpt&meet=1
# 말성적조회. 일 업데이트
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/girok/horse-rslt/20161023dacom21.rpt&meet=1
# 경주마정보. 목일 업데이트
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/horse/20161023sdb1.txt&meet=1
# 기수정보. 목일 업데이트
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/jockey/20161023sdb2.txt&meet=1
# 조교사정보. 목일 업데이트
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/trainer/20161023sdb3.txt&meet=1
race_url = "http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/horse/"
meet = 1  # 1: seoul, 2: jeju, 3: bukyeong
date = datetime.date.today() + datetime.timedelta(days=1)
#date = datetime.date(2015, 5, 25)
while date > datetime.date(2011, 01, 01):
    date += datetime.timedelta(days=-1)
    if date.weekday() not in [3, 6]:
        continue
    date_s = int("%d%02d%02d" % (date.year, date.month, date.day))
    request = "%s%dsdb1.txt&meet=%d" % (race_url, date_s, meet)
    try:
        response_body = urlopen(request).read()
        fout = open("../txt/horse/horse_%d_%d.txt" % (meet, date_s), 'w')
        fout.write(response_body)
        fout.close()
        print "[%s] data is downloaded" % date_s
    except:
        print '[%s] data downloading failed' % date_s

print "job has completed"

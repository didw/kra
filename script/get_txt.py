#!/usr/bin/python
# -*- coding: utf-8 -*-

from urllib2 import Request, urlopen
import datetime
import time
import os

# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/jungbo/rcresult/20161030dacom11.rpt&meet=3
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/horse/20161102cdb1.txt&meet=2
def download_txt(bd, ed, meet):
    data = [# seoul
            [["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/rcresult/", "rcresult", "dacom11.rpt", [5, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/horse/", "horse", "sdb1.txt", [3, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/jockey/", "jockey", "sdb2.txt", [3, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/trainer/", "trainer", "sdb3.txt", [3, 6]]],
            # jeju
            [["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/jeju/jungbo/rcresult/", "rcresult", "dacom11.rpt", [4, 5]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/horse/", "horse", "cdb1.txt", [2, 5]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/jockey/", "jockey", "cdb2.txt", [2, 5]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/trainer/", "trainer", "cdb3.txt", [2, 5]]],
            # busan
            [["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/jungbo/rcresult/", "rcresult", "dacom11.rpt", [4, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/horse/", "horse", "pdb1.txt", [2, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/jockey/", "jockey", "pdb2.txt", [2, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/trainer/", "trainer", "pdb3.txt", [2, 6]]]
            ]

    for line in data[meet-1]:
        race_url = line[0]
        date = bd + datetime.timedelta(days=-1)
        while date < ed:
            date += datetime.timedelta(days=1)
            if date.weekday() not in line[3]:
                continue
            date_s = int("%d%02d%02d" % (date.year, date.month, date.day))
            request = "%s%d%s&meet=%d" % (race_url, date_s, line[2], meet)
            try:
                fname = "../txt/%d/%s/%s_%d_%d.txt" % (meet, line[1], line[1], meet, date_s)
                if os.path.exists(fname):
                    continue
                response_body = urlopen(request).read()
                fout = open(fname, 'w')
                fout.write(response_body)
                fout.close()
                print "[%s] data is downloaded" % request
            except:
                print '[%s] data downloading failed' % request
    print "job has completed"

if __name__ == '__main__':
    download_txt(datetime.date(2011, 1, 1), datetime.date.today(), 3)

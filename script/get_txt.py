#!/usr/bin/python
# -*- coding: utf-8 -*-

from urllib2 import Request, urlopen
import datetime
import time
import os

# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/jungbo/rcresult/20161030dacom11.rpt&meet=3
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/horse/20161102cdb1.txt&meet=2
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/chulma/20161103dacom01.rpt&meet=1
def download_txt(bd, ed, meet):
    data = [# seoul
            [["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/rcresult/", "rcresult", "dacom11.rpt", [5, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/horse/", "horse", "sdb1.txt", [3, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/jockey/", "jockey", "sdb2.txt", [3, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/trainer/", "trainer", "sdb3.txt", [3, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/chulma/", "chulma", "dacom01.rpt", [3]]],
            # jeju
            [["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/jeju/jungbo/rcresult/", "rcresult", "dacom11.rpt", [4, 5]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/horse/", "horse", "cdb1.txt", [2, 5]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/jockey/", "jockey", "cdb2.txt", [2, 5]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/trainer/", "trainer", "cdb3.txt", [2, 5]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/jeju/jungbo/chulma/", "chulma", "dacom01.rpt", [2]]],
            # busan
            [["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/jungbo/rcresult/", "rcresult", "dacom11.rpt", [4, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/horse/", "horse", "pdb1.txt", [2, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/jockey/", "jockey", "pdb2.txt", [2, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/trainer/", "trainer", "pdb3.txt", [2, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/jungbo/chulma/", "chulma", "dacom01.rpt", [2]]]
            ]

    for line in data[meet-1]:
        race_url = line[0]
        date = ed + datetime.timedelta(days=1)
        while date > bd:
            date += datetime.timedelta(days=-1)
            if date.weekday() not in line[3]:
                continue
            date_s = int("%d%02d%02d" % (date.year, date.month, date.day))
            request = "%s%d%s&meet=%d" % (race_url, date_s, line[2], meet)
            try:
                fname = "../txt/%d/%s/%s_%d_%d.txt" % (meet, line[1], line[1], meet, date_s)
                if os.path.exists(fname):
                    break
                response_body = urlopen(request).read()
                fout = open(fname, 'w')
                fout.write(response_body)
                fout.close()
                print "[%s] data is downloaded" % request
            except:
                print '[%s] data downloading failed' % request
    print "job has completed"

def download_dist_rec(bd, ed, meet):
    data = [# seoul http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=1&rcNo=3&rcDate=20070415
        #      http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=1&rcNo=5&rcDate=20070714
            [["http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1", [5, 6]]],
            [["http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1", [4, 5]]],
            [["http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1", [4, 6]]],
    ]
    for line in data[meet-1]:
        race_url = line[0]
        date = ed + datetime.timedelta(days=1)
        while date > bd:
            date += datetime.timedelta(days=-1)
            if date.weekday() not in line[1]:
                continue
            for rcno in range(1, 20):
                date_s = int("%d%02d%02d" % (date.year, date.month, date.day))
                request = "%s&meet=%d&rcNo=%d&rcDate=%d" % (race_url, meet, rcno, date_s)
                try:
                    fname = "../txt/%d/dist_rec/dist_rec_%d_%d_%d.txt" % (meet, meet, date_s, rcno)
                    if os.path.exists(fname):
                        break
                    response_body = urlopen(request).read()
                    fout = open(fname, 'w')
                    fout.write(response_body)
                    fout.close()
                    print "[%s] data is downloaded" % request
                except:
                    print '[%s] data downloading failed' % request
    print "job has completed"


if __name__ == '__main__':
    for i in range(1, 4):
        download_dist_rec(datetime.date(2015, 1, 1), datetime.date.today(), i)
        download_txt(datetime.date(2015, 1, 1), datetime.date.today(), i)


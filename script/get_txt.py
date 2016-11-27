#!/usr/bin/python
# -*- coding: utf-8 -*-

from urllib2 import Request, urlopen
import datetime
import time
import os

# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/jungbo/rcresult/20161030dacom11.rpt&meet=3
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/horse/20161102cdb1.txt&meet=2
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/chulma/20161103dacom01.rpt&meet=1
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/jockey/20070216pdb2.txt&meet=3
def download_txt(bd, ed, meet, overwrite=False):
    data = [# seoul
            [["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/rcresult/", "rcresult", "dacom11.rpt", [5, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/horse/", "horse", "sdb1.txt", [3, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/jockey/", "jockey", "sdb2.txt", [3, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/trainer/", "trainer", "sdb3.txt", [3, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/chulma/", "chulma", "dacom01.rpt", [3]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/sokbo/daily-train/", "daily-train", "dacom55.rpt", [0,1,2,3,4,5,6]]],
            # jeju
            [["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/jeju/jungbo/rcresult/", "rcresult", "dacom11.rpt", [4, 5]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/horse/", "horse", "cdb1.txt", [2, 5]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/jockey/", "jockey", "cdb2.txt", [2, 5]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/trainer/", "trainer", "cdb3.txt", [2, 5]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/jeju/jungbo/chulma/", "chulma", "dacom01.rpt", [2]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/jeju/sokbo/daily-train/", "daily-train", "dacom55.rpt", [0,1,2,3,4,5,6]]],
            # busan
            [["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/jungbo/rcresult/", "rcresult", "dacom11.rpt", [4, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/horse/", "horse", "pdb1.txt", [2, 4, 5, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/jockey/", "jockey", "pdb2.txt", [2, 4, 5, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/trainer/", "trainer", "pdb3.txt", [2, 4, 5, 6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/jungbo/chulma/", "chulma", "dacom01.rpt", [2]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/sokbo/daily-train/", "daily-train", "dacom55.rpt", [0,1,2,3,4,5,6]]]
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
                if not overwrite and os.path.exists(fname):
                    continue
                response_body = urlopen(request).read()
                fout = open(fname, 'w')
                fout.write(response_body)
                fout.close()
                print "[%s] data is downloaded" % request
            except:
                print '[%s] data downloading failed' % request
    print "job has completed"


# 해당거리 전적:    http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=1&rcNo=3&rcDate=20070415
# 출전표:           http://race.kra.co.kr/chulmainfo/chulmaDetailInfoChulmapyo.do?Act=02&Sub=1&meet=1&rcNo=1&rcDate=20161113
# 체중, 최종출전일: http://race.kra.co.kr/chulmainfo/chulmaDetailInfoWeight.do?Act=02&Sub=1&meet=1&rcNo=1&rcDate=20161113
# 훈련현황:         http://race.kra.co.kr/chulmainfo/chulmaDetailInfoTrainState.do?Act=02&Sub=1&meet=1&rcNo=1&rcDate=20161113

def download_chulmaDetailInfo(bd, ed, meet, overwrite=False):
    data = [# seoul http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=1&rcNo=3&rcDate=20070415
        #      http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=1&rcNo=5&rcDate=20070714
            [["DistanceRecord.do?Act=02&Sub=1", "dist_rec", [5, 6]],
             ["Chulmapyo.do?Act=02&Sub=1", "chulmapyo", [5, 6]],
             ["Weight.do?Act=02&Sub=1", "weight", [5, 6]],
             ["TrainState.do?Act=02&Sub=1", "train_state", [5, 6]]
             ],
            [["DistanceRecord.do?Act=02&Sub=1", "dist_rec", [4, 5]],
             ["Chulmapyo.do?Act=02&Sub=1", "chulmapyo", [4, 5]],
             ["Weight.do?Act=02&Sub=1", "weight", [4, 5]],
             ["TrainState.do?Act=02&Sub=1", "train_state", [4, 5]],
             ],
            [["DistanceRecord.do?Act=02&Sub=1", "dist_rec", [4, 6]],
             ["Chulmapyo.do?Act=02&Sub=1", "chulmapyo", [4, 6]],
             ["Weight.do?Act=02&Sub=1", "weight", [4, 6]],
             ["TrainState.do?Act=02&Sub=1", "train_state", [4, 6]],
             ],
    ]
    for line in data[meet-1]:
        base_url = "http://race.kra.co.kr/chulmainfo/chulmaDetailInfo"
        race_url = base_url + line[0]
        date = bd + datetime.timedelta(days=-1)
        while date < ed:
            date += datetime.timedelta(days=1)
            if date.weekday() not in line[2]:
                continue
            for rcno in range(1, 20):
                date_s = int("%d%02d%02d" % (date.year, date.month, date.day))
                request = "%s&meet=%d&rcNo=%d&rcDate=%d" % (race_url, meet, rcno, date_s)
                try:
                    fname = "../txt/%d/%s/%s_%d_%d_%d.txt" % (meet, line[1], line[1], meet, date_s, rcno)
                    if not overwrite and os.path.exists(fname):
                        continue
                    response_body = urlopen(request).read()
                    fout = open(fname, 'w')
                    fout.write(response_body)
                    fout.close()
                    print("[%s] data is downloaded" % request)
                except:
                    print('[%s] data downloading failed' % request)
    print("job has completed")


if __name__ == '__main__':
    for i in range(3, 4):
        download_chulmaDetailInfo(datetime.date(2016, 11, 20), datetime.date.today()+datetime.timedelta(days=-1), i, True)
        download_txt(datetime.date(2016, 11, 20), datetime.date.today()+datetime.timedelta(days=-1), i, True)


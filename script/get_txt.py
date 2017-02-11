#!/usr/bin/python
# -*- coding: utf-8 -*-

from urllib2 import Request, urlopen
import datetime
import time
import os

# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/jungbo/rcresult/20161030dacom11.rpt&meet=3
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/horse/20161102cdb1.txt&meet=2
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/chulma/20161103dacom01.rpt&meet=1
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/sokbo/daily-train/20161113dacom55.rpt&meet=1
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/ap-check-rslt/20141226dacom23.rpt&meet=1
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/sokbo/weekly-clinic/20170101dacom72.rpt&meet=1
# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/sokbo/weekly-jangu/20161222dacom71.rpt&meet=1
def download_txt(bd, ed, meet, overwrite=False):
    data = [# seoul
            [["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/rcresult/", "rcresult", "dacom11.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/horse/", "horse", "sdb1.txt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/jockey/", "jockey", "sdb2.txt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/trainer/", "trainer", "sdb3.txt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/chulma/", "chulma", "dacom01.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/sokbo/daily-train/", "daily-train", "dacom55.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/ap-check-rslt/", "ap-check-rslt", "dacom23.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/sokbo/weekly-clinic/", "weekly-clinic", "dacom72.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/sokbo/weekly-jangu/", "weekly-jangu", "dacom71.rpt", [0,1,2,3,4,5,6]]],
            # jeju
            [["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/jeju/jungbo/rcresult/", "rcresult", "dacom11.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/horse/", "horse", "cdb1.txt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/jockey/", "jockey", "cdb2.txt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/jeju/trainer/", "trainer", "cdb3.txt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/jeju/jungbo/chulma/", "chulma", "dacom01.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/jeju/sokbo/daily-train/", "daily-train", "dacom55.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/jeju/jungbo/ap-check-rslt/", "ap-check-rslt", "dacom23.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/jeju/sokbo/weekly-clinic/", "weekly-clinic", "dacom72.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/jeju/sokbo/weekly-jangu/", "weekly-jangu", "dacom71.rpt", [0,1,2,3,4,5,6]]],
            # busan
            [["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/jungbo/rcresult/", "rcresult", "dacom11.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/horse/", "horse", "pdb1.txt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/jockey/", "jockey", "pdb2.txt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/busan/trainer/", "trainer", "pdb3.txt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/jungbo/chulma/", "chulma", "dacom01.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/sokbo/daily-train/", "daily-train", "dacom55.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/jungbo/ap-check-rslt/", "ap-check-rslt", "dacom23.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/sokbo/weekly-clinic/", "weekly-clinic", "dacom72.rpt", [0,1,2,3,4,5,6]],
             ["http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/busan/sokbo/weekly-jangu/", "weekly-jangu", "dacom71.rpt", [0,1,2,3,4,5,6]]]
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
                if os.path.exists(fname) and os.path.getsize(fname) < 100:
                    os.remove(fname)
                if not overwrite and os.path.exists(fname):
                    continue
                response_body = urlopen(request).read()
                fout = open(fname, 'w')
                fout.write(response_body)
                fout.close()
                if os.path.getsize(fname) < 100:
                    os.remove(fname)
                print "[%s] data is downloaded" % request
            except KeyError:
                print '[%s] data downloading failed' % request
                print 'or fail to save %s' % fname
    print "job has completed"


# 해당거리 전적:    http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=1&rcNo=3&rcDate=20070415
#                   http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=2&rcNo=1&rcDate=20161119
#                   http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=2&rcNo=2&rcDate=20161119
# 출전표:           http://race.kra.co.kr/chulmainfo/chulmaDetailInfoChulmapyo.do?Act=02&Sub=1&meet=1&rcNo=1&rcDate=20161113
# 체중, 최종출전일: http://race.kra.co.kr/chulmainfo/chulmaDetailInfoWeight.do?Act=02&Sub=1&meet=1&rcNo=1&rcDate=20161113
# 훈련현황:         http://race.kra.co.kr/chulmainfo/chulmaDetailInfoTrainState.do?Act=02&Sub=1&meet=1&rcNo=1&rcDate=20161113

def download_chulmaDetailInfo(bd, ed, meet, overwrite=False):
    data = [# seoul http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=1&rcNo=3&rcDate=20070415
        #      http://race.kra.co.kr/chulmainfo/chulmaDetailInfoDistanceRecord.do?Act=02&Sub=1&meet=1&rcNo=5&rcDate=20070714
            [["DistanceRecord.do?Act=02&Sub=1", "dist_rec", [5, 6], 31700],
             ["Chulmapyo.do?Act=02&Sub=1", "chulmapyo", [5, 6], 33400],
             ["Weight.do?Act=02&Sub=1", "weight", [5, 6], 31400],
             ["TrainState.do?Act=02&Sub=1", "train_state", [5, 6], 32000]
             ],
            [["DistanceRecord.do?Act=02&Sub=1", "dist_rec", [4, 5], 31700],
             ["Chulmapyo.do?Act=02&Sub=1", "chulmapyo", [4, 5], 33400],
             ["Weight.do?Act=02&Sub=1", "weight", [4, 5], 31400],
             ["TrainState.do?Act=02&Sub=1", "train_state", [4, 5], 32000],
             ],
            [["DistanceRecord.do?Act=02&Sub=1", "dist_rec", [4, 6], 31700],
             ["Chulmapyo.do?Act=02&Sub=1", "chulmapyo", [4, 6], 33400],
             ["Weight.do?Act=02&Sub=1", "weight", [4, 6], 31400],
             ["TrainState.do?Act=02&Sub=1", "train_state", [4, 6], 32000],
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
                    if os.path.getsize(fname) < line[3]:
                        os.remove(fname)
                    print("[%s] data is downloaded" % request)
                except:
                    print('[%s] data downloading failed' % request)
    print("job has completed")

def download_racehorse(hrno_b, hrno_e, meet, overwrite=False):
    data = [# seoul http://race.kra.co.kr/racehorse/profileRaceScore.do?Act=02&Sub=1&meet=1&hrNo=040000
            ["profileRaceScore.do?Act=02&Sub=1&", "racehorse",
             ],
            ["profileRaceScore.do?Act=02&Sub=1&", "racehorse",
             ],
            ["profileRaceScore.do?Act=02&Sub=1&", "racehorse",
             ]
    ]
    line = data[meet-1]
    base_url = "http://race.kra.co.kr/racehorse/"
    race_url = base_url + line[0]
    for hrno in range(hrno_b, hrno_e):
        request = "%s&meet=%d&hrNo=%06d" % (race_url, meet, hrno)
        try:
            fname = "../txt/%d/%s/%s_%d_%06d.txt" % (meet, line[1], line[1], meet, hrno)
            if not overwrite and os.path.exists(fname):
                continue
            response_body = urlopen(request).read()
            fout = open(fname, 'w')
            fout.write(response_body)
            fout.close()
            if os.path.getsize(fname) < 31100:
                os.remove(fname)
            print("[%s] data is downloaded" % request)
        except:
            print('[%s] data downloading failed' % request)
    print("job has completed")


if __name__ == '__main__':
    for i in range(1, 2):
        #download_racehorse(29301, 29301, i, False)
        download_chulmaDetailInfo(datetime.date(2017, 2, 1), datetime.date.today(), i, False)
        download_txt(datetime.date(2017, 2, 1), datetime.date.today(), i, False)


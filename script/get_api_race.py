#!/usr/bin/python
# -*- coding: utf-8 -*-

from urllib2 import Request, urlopen
from urllib import urlencode, quote_plus

# http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/rcresult/20161016dacom11.rpt&meet=1
race_url = "http://data.kra.co.kr/publicdata/service/race/getRace"
service_key = "MZBYd4tuPPcTF%2Flqt01Rco4IPTC3r5SZDRbDnoW5P7XG3aCIMMGepC0D%2FnKo1Yu5OVyDYjcAk9l3qg34t6XGzA%3D%3D"
meet = 2  # 1: seoul, 2: jeju, 3: bukyeong

for year in range(1993, 1900, -1):
    for month in range(12, 0, -1):
        date = int("%d%02d" % (year, month))
        url = "%s?ServiceKey=%s&meet=%d&rcDate=%d" % (race_url, service_key, meet, date)
        request = Request(url)
        request.get_method = lambda: 'GET'
        response_body = urlopen(request).read()
        fout = open("../xml/race/getrace_%d_%d.xml" % (meet, date), 'w')
        fout.write(response_body)
        fout.close()
        print "%d is downloaded" % date

print "all completed"

#!/usr/bin/python
# -*- coding: utf-8 -*-

from urllib2 import Request, urlopen
from urllib import urlencode, quote_plus
race_url = "http://data.kra.co.kr/publicdata/service/entry/getEntry"
service_key = "MZBYd4tuPPcTF%2Flqt01Rco4IPTC3r5SZDRbDnoW5P7XG3aCIMMGepC0D%2FnKo1Yu5OVyDYjcAk9l3qg34t6XGzA%3D%3D"
meet = 1  # 1: seoul, 2: jeju, 3: bukyeong

year = 2016
month = 10
date = int("%d%02d" % (year, month))
url = "%s?ServiceKey=%s&meet=%d&rcDate=%d" % (race_url, service_key, meet, date)
request = Request(url)
request.get_method = lambda: 'GET'
response_body = urlopen(request).read()
fout = open("../xml/entry/get_entry_%d_%d.xml" % (meet, date), 'w')
fout.write(response_body)
fout.close()
print "%d is downloaded" % date
print "all completed"

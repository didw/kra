#!/usr/bin/python
# -*- coding: utf-8 -*-

from urllib2 import Request, urlopen
from urllib import urlencode, quote_plus

def get_data(meet, date):
    race_url = "http://data.kra.co.kr/publicdata/service/entry/getEntry"
    service_key = "MZBYd4tuPPcTF%2Flqt01Rco4IPTC3r5SZDRbDnoW5P7XG3aCIMMGepC0D%2FnKo1Yu5OVyDYjcAk9l3qg34t6XGzA%3D%3D"
    url = "%s?ServiceKey=%s&meet=%d&rcDate=%d" % (race_url, service_key, meet, date/100)
    request = Request(url)
    request.get_method = lambda: 'GET'
    response_body = urlopen(request).read()
    fout = open("../xml/entry/get_entry_%d_%d.xml" % (meet, date/100), 'w')
    fout.write(response_body)
    fout.close()
    print "entry(%d) is downloaded" % date

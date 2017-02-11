#!/usr/bin/python
# -*- coding: utf-8 -*-

from urllib2 import Request, urlopen

def get_data(meet):
    race_url = "http://data.kra.co.kr/publicdata/service/hr/getHR"
    service_key = "MZBYd4tuPPcTF%2Flqt01Rco4IPTC3r5SZDRbDnoW5P7XG3aCIMMGepC0D%2FnKo1Yu5OVyDYjcAk9l3qg34t6XGzA%3D%3D"
    url = "%s?meet=%d&ServiceKey=%s" % (race_url, meet, service_key)
    request = Request(url)
    request.get_method = lambda: 'GET'
    response_body = urlopen(request).read()
    fout = open("../xml/getHR_%d.xml" % (meet), 'w')
    fout.write(response_body)
    fout.close()
    print "hr is downloaded"

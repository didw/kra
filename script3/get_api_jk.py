#!/usr/bin/python
# -*- coding: utf-8 -*-

import requests


def get_data(meet):
    race_url = "http://data.kra.co.kr/publicdata/service/jk/getJK"
    service_key = "MZBYd4tuPPcTF%2Flqt01Rco4IPTC3r5SZDRbDnoW5P7XG3aCIMMGepC0D%2FnKo1Yu5OVyDYjcAk9l3qg34t6XGzA%3D%3D"
    url = "%s?meet=%d&ServiceKey=%s" % (race_url, meet, service_key)
    response_body = requests.get(url)
    fout = open("../xml/getJK_%d.xml" % (meet), 'w')
    fout.write(response_body.text)
    fout.close()
    print("jk is downloaded")

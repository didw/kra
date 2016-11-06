#!/usr/bin/python
# -*- coding: utf-8 -*-

from urllib2 import Request
from bs4 import BeautifulSoup
import pandas as pd
from urllib2 import urlopen

def get_data(meet, date):
    race_url = "http://data.kra.co.kr/publicdata/service/entry/getEntry"
    service_key = "" # https://www.kra.co.kr/openDataPublic.do 에서 인증키를 신청해야함
    url = "%s?ServiceKey=%s&meet=%d&rcDate=%d" % (race_url, service_key, meet, date)
    request = Request(url)
    request.get_method = lambda: 'GET'
    response_body = urlopen(request).read()
    fout = open("../xml/entry/get_entry_%d_%d.xml" % (meet, date), 'w')
    fout.write(response_body)
    fout.close()
    print "entry(%d) is downloaded" % date



def parse_xml_entry(meet, date):
    data = []
    filename = '../get_entry_%d_%d.xml' % (meet, date)
    file_input = open(filename)
    print "process in %s" % filename
    response_body = file_input.read()
    xml_text = BeautifulSoup(response_body, 'html.parser')

    for itemElm in xml_text.findAll('item'):
        adata = [itemElm.rcdist.string,
                 itemElm.chulno.string,
                 itemElm.hrname.string,
                 itemElm.prdctyname.string,
                 itemElm.age.string,
                 itemElm.wgbudam.string,
                 itemElm.jkname.string,

                 itemElm.trname.string,
                 itemElm.owname.string,
                 itemElm.rcno.string,

                 itemElm.cntt.string,
                 itemElm.ord1t.string,
                 itemElm.ord2t.string,
                 itemElm.cnty.string,
                 itemElm.ord1y.string,
                 itemElm.ord2y.string,
                 ]
        data.append(adata)
    return pd.DataFrame(data)

if __name__ == '__main__':
    get_data(1, 20161106)
    print(parse_xml_entry(1, 20161106))


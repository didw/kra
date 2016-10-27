#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob


def parse_xml_entry(meet, date):
    data = []
    filename = 'data/get_entry_%s_%s.xml' % (meet, date)
    file_input = open(filename)
    print "process in %s" % filename
    response_body = file_input.read()
    xml_text = BeautifulSoup(response_body, 'html.parser')
    for itemElm in xml_text.findAll('item'):
        print itemElm
        try:
            data.append([itemElm.hrname.string,
                         itemElm.age.string,
                         itemElm.budam.string,
                         itemElm.calt.string,
                         itemElm.caly.string,
                         itemElm.cal_6m.string,
                         itemElm.chulno.string,
                         itemElm.cntt.string,
                         itemElm.cnty.string,
                         itemElm.jkname.string,
                         itemElm.meet.string,
                         itemElm.ord1t.string,
                         itemElm.ord1y.string,
                         itemElm.ord2t.string,
                         itemElm.ord2y.string,
                         itemElm.owname.string,
                         itemElm.prdctyname.string,
                         itemElm.rank.string,
                         itemElm.rcdate.string,
                         itemElm.rcdist.string,
                         itemElm.rcname.string,
                         itemElm.rcno.string,
                         itemElm.sttime.string,
                         itemElm.trname.string,
                         itemElm.wgbudam.string])
        except:
            pass

    df = pd.DataFrame(data)
    df.columns = ["hrname", "age", "budam", "calT", "calY", "cal_6m", "chulNo", "cntT", "cntY", "jkName",
                    "meet", "ord1T", "ord1Y", "ord2T", "ord2Y", "owName", "prdCtyNam", "rank", "rcDate", "rcDist",
                    "rcName", "rcNo", "stTime", "trName", "wgBudam"]
    return df

if __name__ == 'main':
    meet = 1
    date = 201610
    data = parse_xml_entry(meet, date)
    print data

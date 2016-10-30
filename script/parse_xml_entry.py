#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob
import re
from urllib2 import urlopen
import parse_xml_jk as xj
import parse_xml_hr as xh
import parse_xml_tr as xt



def get_humidity():
    url = "http://race.kra.co.kr/chulmainfo/trackView.do?Act=02&Sub=10&meet=1"
    response_body = urlopen(url).read()
    line = unicode(response_body, 'euc-kr').encode('utf-8')
    print("%s" % line)
    p = re.compile(unicode(r'(?<=함수율 <span>: )\d+(?=\%\()', 'utf-8').encode('utf-8'), re.MULTILINE)
    pl = p.search(line)
    res = 10
    if pl is not None:
        res = pl.group()
    return res


def parse_xml_entry(meet, date):
    data = []
    filename = '../xml/entry/get_entry_%s_%s.xml' % (meet, date)
    file_input = open(filename)
    print "process in %s" % filename
    response_body = file_input.read()
    xml_text = BeautifulSoup(response_body, 'html.parser')
    humidity = get_humidity()
    for itemElm in xml_text.findAll('item'):
        print itemElm
        try:
            """['course', 'humidity', 'rank', 'idx', 'name', 'cntry', 'gender', 'age', 'budam', 'jockey', \
              'trainer', 'owner', 'weight', 'dweight', 'rctime', 'r1', 'r2', 'r3', 'hr_days', 'hr_t1', \
              'hr_t2', 'hr_y1', 'hr_y2', 'jk_t1', 'jk_t2', 'jk_y1', 'jk_y2', 'tr_t1', 'tr_t2', 'tr_y1', \
              'tr_y2']
            """
            data.append([itemElm.rcdist.string,
                         humidity,
                         itemElm.chulno.string,
                         itemElm.hrname.string,
                         itemElm.prdctyname.string,

                         itemElm.age.string,
                         itemElm.budam.string,
                         itemElm.calt.string,
                         itemElm.caly.string,
                         itemElm.cal_6m.string,
                         itemElm.cntt.string,
                         itemElm.cnty.string,
                         itemElm.jkname.string,
                         itemElm.meet.string,
                         itemElm.ord1t.string,
                         itemElm.ord1y.string,
                         itemElm.ord2t.string,
                         itemElm.ord2y.string,
                         itemElm.owname.string,
                         itemElm.rank.string,
                         itemElm.rcdate.string,
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


if __name__ == '__main__':
    meet = 1
    date = 201610
    data = parse_xml_entry(meet, date)
    print data

#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob


def parse_xml_jk(meet):
    data = []
    filename  = '../xml/getJK_%d.xml' % meet
    file_input = open(filename)
    print("process in %s" % filename)
    response_body = file_input.read()
    xml_text = BeautifulSoup(response_body, 'html.parser')
    for itemElm in xml_text.findAll('item'):
        #print itemElm
        try:
            data.append([unicode(itemElm.birth.string),
                         unicode(itemElm.cntt.string),
                         unicode(itemElm.cnty.string),
                         unicode(itemElm.jkname.string),
                         unicode(itemElm.ord1t.string),
                         unicode(itemElm.ord1y.string),
                         unicode(itemElm.ord2t.string),
                         unicode(itemElm.ord2y.string),
                         unicode(itemElm.part.string),
                         unicode(itemElm.stdate.string),
                         unicode(itemElm.wgother.string),
                         unicode(itemElm.wgpart.string)])
        except:
            pass

    df = pd.DataFrame(data)
    df.columns = ["birth", "cntT", "cntY", "jkName", "ord1T", "ord1Y", "ord2T", "ord2Y", "part", "stDate", "wgOther", "wgPart"]
    return df


if __name__ == 'main':
    meet = 1
    data = parse_xml_jk(meet)
    print(data)

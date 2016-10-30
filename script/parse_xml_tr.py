#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob


def parse_xml_tr(meet):
    data = []
    filename  = '../xml/getTR_%d.xml' % meet
    file_input = open(filename)
    print "process in %s" % filename
    response_body = file_input.read()
    xml_text = BeautifulSoup(response_body, 'html.parser')
    for itemElm in xml_text.findAll('item'):
        #print itemElm
        try:
            data.append([itemElm.birth.string,
                        itemElm.cntt.string,
                        itemElm.cnty.string,
                        itemElm.trname.string,
                        itemElm.ord1t.string,
                        itemElm.ord1y.string,
                        itemElm.ord2t.string,
                        itemElm.ord2y.string,
                        itemElm.part.string,
                        itemElm.stdate.string])
        except:
            pass

    df = pd.DataFrame(data)
    df.columns = ["birth", "cntT", "cntY", "trName", "ord1T", "ord1Y", "ord2T", "ord2Y", "part", "stDate"]
    return df

if __name__ == 'main':
    meet = 1
    data = parse_xml_tr(meet)
    print(data)

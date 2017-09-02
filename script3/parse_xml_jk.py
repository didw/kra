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
            data.append([(itemElm.birth.string),
                         (itemElm.cntt.string),
                         (itemElm.cnty.string),
                         (itemElm.jkname.string),
                         (itemElm.ord1t.string),
                         (itemElm.ord1y.string),
                         (itemElm.ord2t.string),
                         (itemElm.ord2y.string),
                         (itemElm.part.string),
                         (itemElm.stdate.string),
                         (itemElm.wgother.string),
                         (itemElm.wgpart.string)])
        except NameError:
            print("NameError")
            raise

    df = pd.DataFrame(data)
    df.columns = ["birth", "cntT", "cntY", "jkName", "ord1T", "ord1Y", "ord2T", "ord2Y", "part", "stDate", "wgOther", "wgPart"]
    return df


if __name__ == 'main':
    meet = 1
    data = parse_xml_jk(meet)
    print(data)

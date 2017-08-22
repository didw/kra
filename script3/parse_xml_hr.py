#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob
import codecs


def parse_xml_hr(meet):
    data = []
    filename  = '../xml/getHR_%d.xml' % meet
    file_input = codecs.open(filename, "r", "utf-8")
    print("process in %s" % filename)
    response_body = file_input.read()
    xml_text = BeautifulSoup(response_body, 'html.parser')
    for itemElm in xml_text.findAll('item'):
        data.append([itemElm.birth.string,
                     itemElm.cntt.string,
                     itemElm.cnty.string,
                     itemElm.hrname.string.replace('★', ''),
                     itemElm.ord1t.string,
                     itemElm.ord1y.string,
                     itemElm.ord2t.string,
                     itemElm.ord2y.string,
                     itemElm.sex.string])

    df = pd.DataFrame(data)
    df.columns = ["birth", "cntT", "cntY", "hrName", "ord1T", "ord1Y", "ord2T", "ord2Y", "gender"]
    return df

if __name__ == '__main__':
    meet = 1
    data = parse_xml_hr(meet)
    print(data)

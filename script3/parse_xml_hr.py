#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob


def parse_xml_hr(meet):
    data = []
    filename  = '../xml/getHR_%d.xml' % meet
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
                         unicode(itemElm.hrname.string).replace('★', ''),
                         unicode(itemElm.ord1t.string),
                         unicode(itemElm.ord1y.string),
                         unicode(itemElm.ord2t.string),
                         unicode(itemElm.ord2y.string),
                         unicode(itemElm.sex.string)])
        except:
            pass

    df = pd.DataFrame(data)
    df.columns = ["birth", "cntT", "cntY", "hrName", "ord1T", "ord1Y", "ord2T", "ord2Y", "gender"]
    return df

if __name__ == '__main__':
    meet = 1
    data = parse_xml_hr(meet)
    print(data)

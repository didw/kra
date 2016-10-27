#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob

data = []
filename  = 'data/getTR_1.xml'
file_input = open(filename)
print "process in %s" % filename
response_body = file_input.read()
xml_text = BeautifulSoup(response_body, 'html.parser')
for itemElm in xml_text.findAll('item'):
    print itemElm
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

print data
data = pd.DataFrame(data)
data.columns = ["birth", "cntT", "cntY", "jkName", "ord1T", "ord1Y", "ord2T", "ord2Y", "part", "stDate"]
print(data)

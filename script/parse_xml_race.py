#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob

data = []
for filename in glob.glob('data/*'):
    file_input = open(filename)
    print "process in %s" % filename
    response_body = file_input.read()
    xml_text = BeautifulSoup(response_body, 'html.parser')
    for itemElm in xml_text.findAll('item'):
        try:
            data.append([itemElm.hrname.string,
                        itemElm.age.string,
                        itemElm.chulno.string,
                        itemElm.corner.string,
                        itemElm.differ.string,
                        itemElm.jkname.string,
                        itemElm.meet.string,
                        itemElm.ord.string,
                        itemElm.owname.string,
                        itemElm.plc.string,
                        itemElm.prdctynm.string,
                        itemElm.rcdate.string,
                        itemElm.rcno.string,
                        itemElm.rctime.string,
                        itemElm.plc.string,
                        itemElm.sex.string,
                        itemElm.trname.string,
                        itemElm.wgbudam.string,
                        itemElm.wghr.string,
                        itemElm.win.string])
        except:
            pass

data = pd.DataFrame(data)
data.columns = ["name", "age", "chulno", "corner", "differ", "jkname", "meet", "ord", "owname", "plc",
                   "prdctynm", "rcdate", "rcno", "rctime", "plc", "sex", "trname", "wgbudam", "wghr", "win"]
print(data)

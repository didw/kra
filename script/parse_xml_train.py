#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import pandas as pd
import glob


def parse_xml_train(date, meet):
    data = []
    filename  = '../xml/getTrain_%d_%d.xml' % (date, meet)
    file_input = open(filename)
    print "process in %s" % filename
    response_body = file_input.read()
    xml_text = BeautifulSoup(response_body, 'html.parser')
    for itemElm in xml_text.findAll('item'):
        print itemElm
        try:
            data.append([itemElm.hrname.string,
                        itemElm.wghr.string])
        except:
            pass

    df = pd.DataFrame(data)
    df.columns = ["hrName", "weight"]
    return df


if __name__ == '__main__':
    date = 201610
    meet = 1
    data = parse_xml_train(date, meet)
    print data

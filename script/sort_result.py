#!/usr/bin/python
# -*- coding: utf-8 -*-

import re

def dict_test(fname, rcno=1):

    data_ss = {}
    data_sb = {}
    fInput = open(fname)
    first = True
    while True:
        if first or re.search(r'rcno: %d,'%rcno, line) is None:
            line = fInput.readline()
            first = False
        if line == None or len(line) == 0:
            break
        if re.search(r'rcno: %d,'%rcno, line) is not None:
            line = fInput.readline()
            while re.search(r'===', line) is None:
                line = fInput.readline()
                if line == None or len(line) == 0:
                    break
                line_parsing = re.search(r'\d+,\d+,\d+', line)
                if line_parsing is None:
                    continue
                key = line_parsing.group()
                if re.search(r'(?<=ss: )\d+', line) is not None:
                    value = int(re.search(r'(?<=ss: )\d+', line).group())
                    try:
                        data_ss[key] += value
                    except KeyError:
                        data_ss[key] = value
                if re.search(r'(?<=sb: )\d+', line) is not None:
                    key2 = "%s"%sorted(key.split(','))
                    value = int(re.search(r'(?<=sb: )\d+', line).group())
                    try:
                        data_sb[key2] += value
                    except KeyError:
                        data_sb[key2] = value
    print("ss")
    for k,v in sorted(data_ss.items()):
        print(k, v)
    print("sb")
    for k,v in sorted(data_sb.items()):
        print(k, v)

if __name__ == '__main__':
    fname = '../result/1703/12_0.txt'
    for i in range(1,13):
        print("===%d==="%i)
        dict_test(fname, i)

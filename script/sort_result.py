#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import pickle

def make_dict(fname, rcno=1):
    data_ss = {}
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
                key = tuple(map(int, line_parsing.group().split(',')))
                if re.search(r'(?<=ss: )\d+', line) is not None:
                    value = int(re.search(r'(?<=ss: )\d+', line).group())
                    try:
                        data_ss[key] += value
                    except KeyError:
                        data_ss[key] = value
    print("ss")
    for k,v in sorted(data_ss.items()):
        v = int((v+50)/100)*100
        print(k, v)
    return data_ss

if __name__ == '__main__':
    init_day = 19
    res_dict = {"Sat": {}, "Sun": {}}
    Day_list = ["Sat", "Sun"]
    for day in [init_day, init_day+1]:
        fname = '../result/1708/%d_1.txt' % day
        for i in range(0,16):
            print("===%d==="%i)
            rcno_dict =  make_dict(fname, i)
            if len(rcno_dict) > 0:
                res_dict[Day_list[day-init_day]][i] = rcno_dict
    pickle.dump(res_dict, open("../result/1708/%d.pkl" % init_day, 'wb'))

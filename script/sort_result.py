#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import pickle

def make_dict(day, rcno=1):
    first = True
    data_ss = {}
    for idx in [1,2,3,4]:
        fname = '../result/1709/%d_%d.txt' % (day, idx)
        try:
            fInput = open(fname)
        except IOError:
            print("No file exists: %s" % fname)
            continue
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
    return data_ss

if __name__ == '__main__':
    init_day = 16
    res_dict = {"Sat": {}, "Sun": {}}
    Day_list = ["Sat", "Sun"]
    for day in [init_day, init_day+1]:
        for i in range(0,16):
            rcno_dict =  make_dict(day, i)
            if len(rcno_dict) > 0:
                res_dict[Day_list[day-init_day]][i] = rcno_dict
    print("ss")
    for day in Day_list:
        for rcno in range(0,16):
            if rcno not in res_dict[day]:
                continue
            print("===%d==="%rcno)
            for k,v in sorted(res_dict[day][rcno].items()):
                v = int((v+499)/500)*500
                res_dict[day][rcno][k] = v
                print(k, v)
    pickle.dump(res_dict, open("../result/1709/%d.pkl" % init_day, 'wb'))

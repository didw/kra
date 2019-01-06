import glob
import re
import numpy as np
import os
import codecs

flist = glob.glob('txt/1/rcresult/*txt')


GD = False
GS = False
GSB = False
GSS = True
d = []
s = []
sb = []
ss = []
for fname in flist:
    bdr = False
    nh = False
    today = int(os.path.basename(fname)[-12:-4])
    if today < 20160700: continue
    print("process {}".format(os.path.basename(fname)[-12:-4]))
    with codecs.open(fname, 'r', 'euc-kr') as f:
        body = f.readlines()
    for line in body:
        if len(line)<5: continue
        if '기수명' in line:
            nh = True
            total_horse = 0
            continue
        elif '마 체 중' in line:
            nh = False
            continue
        if '배당률' in line and '단' in line:
            bdr = True
        if '펄' in line:
            bdr = False
        if nh is True:
            if '-' in line: continue
            try:
                total_horse = max(total_horse, int(re.match(r' +\d+', line).group()))
            except AttributeError as e :
                print(e)
                print(line)
        if bdr is True:
            if '단' in line and GD:
                try:
                    a = float(re.search(r'\d+\.\d', line).group())
                    num_possible = total_horse
                    res = a/num_possible
                    d.append(res)
                    print("d appended {}/{}={}, {}".format(a,num_possible,100*res, 100*np.mean(d)))
                except AttributeError as e:
                    print(e)
                    print(line)
                    continue
            if ' 쌍' in line and GS:
                try:
                    a = float(re.search(r'\d+\.\d', line).group())
                    num_possible = total_horse*(total_horse-1)
                    res = a/num_possible
                    s.append(res)
                    print("s appended {}/{}={}, {}".format(a,num_possible,100*res, 100*np.mean(s)))
                except AttributeError as e:
                    print(e)
                    print(line)
                    continue
            if '삼복' in line and GSB:
                try:
                    a = float(re.search(r'\d+\.\d', line).group())
                    num_possible = total_horse*(total_horse-1)*(total_horse-2)/6
                    res = a/num_possible
                    sb.append(res)
                    print("sb appended {}/{}={}, {}".format(a,num_possible,100*res, 100*np.mean(sb)))
                except AttributeError as e:
                    print(e)
                    print(line)
                    continue
            if '삼쌍' in line and GSS:
                try:
                    a = float(re.search(r'\d+\.\d', line).group())
                    num_possible = total_horse*(total_horse-1)*(total_horse-2)
                    res = a/num_possible
                    ss.append(res)
                    print("[{}] ss appended {}/{}={}, {}".format(today,a,num_possible,100*res, 100*np.mean(ss)))
                except AttributeError as e:
                    print(e)
                    print(line)
                    continue
    
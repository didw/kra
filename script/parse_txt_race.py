# -*- coding:utf-8 -*-
from __future__ import print_function
import re
import pandas as pd
import numpy as np


HEADER = re.compile(r'기수명')
WEIGHT = re.compile(r'마 체 중')
WORD = re.compile(r"[^ \n]+")

afile = open('../data/race/20161023dacom11.rpt')

# skip header
for i in range(10):
    line = afile.readline()
    line = unicode(line, 'euc-kr').encode('utf-8')
    if HEADER.search(line) is not None:
        break

# 순위 마번    마    명      산지   성별 연령 부담중량 기수명 조교사   마주명           레이팅
data = []
for i in range(100):
    line = afile.readline()
    line = unicode(line, 'euc-kr').encode('utf-8')
    if line[0] == '-':
        continue
    if WEIGHT.search(line) is not None:
        break

    adata = []
    words = WORD.findall(line)
    print("\nline: %s" % line)
    for w in words:
        print(str(w), end='\t')
        adata.append(str(w))
    data.append(adata)


# 순위 마번    마      명    마 체 중 기  록  위  차 S1F-1C-2C-3C-4C-G1F

for i in range(len(data)):
    print(data[i])

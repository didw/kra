# -*- coding: utf-8 -*-

import parse_txt_race as pr
import datetime
import pandas as pd
import os.path
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.externals import joblib
import random
import numpy as np
import re

# 1 win
def simulation1(pred, ans, target=1):
    #print(ans)
    i = 0
    res1 = 0
    bet = 10.0
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r1 = [float(ans['r1'][i])]
        rcno = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        hrname = ans['name'][i]
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        price = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno and int(ans['rank'][i]) != 1:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            r1.append(float(ans['r1'][i]))
            price = int(ans['price'][i])
            total += 1
            i += 1
        a = price*0.8 / r1[0]
        r1[0] = (price+100000)*0.8 / (a+100000)
        if r1[0]*bet > 200:
            r1[0] *= 0.8
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()
        if len(sim_data) < target:
            continue
        if top[0] == target:
            res1 += bet * (r1[0] - 1)
        else:
            res1 -= bet
        #print("단승식: %f, %f" % (res1, res2))
    return res1


# 1 win in 2-3
def simulation2(pred, ans, target=1):
    i = 0
    res1 = 0
    rcno = 0
    bet = 10.0
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        rcno += 1
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r = float(ans['r2'][i])
        price = int(ans['price'][i])
        a = price*0.8 / r
        r2 = [(price+100000)*0.8 / (a+100000) - 1]
        rc_no = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        price = 0
        while i < len(pred) and int(ans['rcno'][i]) == rc_no and int(ans['rank'][i]) != 1:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            price = int(ans['price'][i])
            r = float(ans['r2'][i])
            a = price*0.8 / r
            r = (price+100000)*0.8 / (a+100000)
            r2.append(r-1)
            total += 1
            i += 1
        # if rack_data or total < total_player:
        #     continue

        if len(sim_data) < 3:
            continue
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()

        if total_player > 7:
            if target == top[0]:
                res1 += bet * r2[0]
            elif target == top[1]:
                res1 += bet * r2[1]
            elif target == top[2]:
                res1 += bet * r2[2]
            else:
                res1 -= bet
        else:
            if target == top[0]:
                res1 += bet * r2[0]
            elif target == top[1]:
                res1 += bet * r2[1]
            else:
                res1 -= bet

        #print("연승식: %f" % (res1))
    return res1

# 2 win
def simulation3(pred, ans, targets=[[1,2],[1,3],[2,3]]):
    bet = 10.0 / len(targets)
    i = 0
    res1 = 0
    assert len(pred) == len(ans)
    while True:
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r3 = float(ans['r3'][i])
        rcno = int(ans['rcno'][i])
        price = int(ans['price'][i])
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        price = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno and int(ans['rank'][i]) != 1:
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            price = int(ans['price'][i])
            total += 1
            i += 1
        # if rack_data or total < total_player:
        #     continue
        price *= 10
        a = price*0.7 / r3
        r3 = (price+100000)*0.7 / (a+100000)
        if r3*bet > 200:
            r3 *= 0.8
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()
        if total < 2:
            continue
        for target in targets:
            if (top[0] in target) and (top[1] in target):
                res1 += bet * (r3-1)
            else:
                res1 -= bet

        #print("복승식: %f" % (res1))
    return res1


def get_num(line, bet):
    num_circle_list = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭"
    try:
        a = num_circle_list.find(line[:3]) / 3 + 1
        b = num_circle_list.find(line[3:6]) / 3 + 1
        r = float(re.search(r'[\d.]+', line[6:]).group()) -1
        if r * bet > 200:
            r *= 0.8
        return [a, b, r]
    except ValueError:
        return [-1, -1, -1]
    except AttributeError:
        return [-1, -1, -1]


# 2 win in 3
def simulation4(pred, ans, target=[1,2]):
    bet = 10.0
    i = 0
    res1 = 0
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r4 = [get_num(ans['bokyeon1'][i], bet)]
        r4.append(get_num(ans['bokyeon2'][i], bet))
        r4.append(get_num(ans['bokyeon3'][i], bet))
        str_rate = ans['bokyeon1'][i] + ans['bokyeon2'][i] + ans['bokyeon3'][i]
        rcno = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        idxes = [int(ans['idx'][i])]
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        price = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno and int(ans['rank'][i]) != 1:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            idxes.append(ans['idx'][i])
            total_player = int(ans['cnt'][i])
            price = int(ans['price'][i])
            total += 1
            i += 1
        # if rack_data or total < total_player:
        #     continue
        if -1 in r4[0] or -1 in r4[1] or -1 in r4[2] or len(sim_data) < 5:
            continue
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()
        succeed = False
        for r in r4:
            if idxes[0] in [r[0], r[1]] and idxes[1] in [r[0], r[1]]:
                if top[0] in target and top[1] in target:
                    res1 += bet*r[2]
                    succeed = True
            elif idxes[0] in [r[0], r[1]] and idxes[2] in [r[0], r[1]]:
                if top[0] in target and top[2] in target:
                    res1 += bet*r[2]
                    succeed = True
            elif idxes[1] in [r[0], r[1]] and idxes[2] in [r[0], r[1]]:
                if top[1] in target and top[2] in target:
                    res1 += bet*r[2]
                    succeed = True
        if not succeed:
            res1 -= bet

        #print("bokyeon: %f" % (res1))
    return res1


import re
# 2 straight win
def simulation5(pred, ans, targets=[[1,2],[1,3],[2,1],[2,3]]):
    bet = 10.0 / len(targets)
    i = 0
    res1 = 0
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r5 = float(re.search(r'\d+\.\d', str(ans['ssang'][i])).group()) - 1
        if r5 == 0:
            r5 = 0.1
        rcno = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        price = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno and int(ans['rank'][i]) != 1:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            price = int(ans['price'][i])
            total += 1
            i += 1
        # if rack_data or total < total_player:
        #     continue
        price *= 10
        a = price*0.7 / r5
        r5 = (price+100000)*0.7 / (a+100000)
        sim_data = pd.Series(sim_data)
        if bet*r5 > 2000:
            r5 *= 0.8
        top = sim_data.rank()
        if total < 2:
            continue
        for target in targets:
            if top[0] == target[0] and top[1] == target[1]:
                res1 += bet * r5
            else:
                res1 -= bet

        #print("ssang: %f" % (res1))
    return res1


# 3 straight win
def simulation6(pred, ans, targets=[[1,2,3], [1,2,4], [1,2,5], [1,3,4], [1,3,5], [1,4,5], [2,3,4], [2,3,5], [2,4,5], [3,4,5]]):
    bet = 10.0 / len(targets)
    i = 0
    res1 = 0
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        try:
            r6 = float(ans['sambok'][i]) - 1
        except TypeError:
            r6 = -1
        except ValueError:
            r6 = float(re.search(r'[\d.]+', ans['sambok'][i]).group()) - 1
        rcno = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        price = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno and int(ans['rank'][i]) != 1:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            price = int(ans['price'][i])
            total += 1
            i += 1
        
        price *= 30
        a = price*0.7 / r6
        r6 = (price+bet)*0.7 / (a+bet)

        if r6 * bet > 200:
            r6 *= 0.8
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()
        if total < 5 or r6 == -1:
            continue
        
        for target in targets:
            if top[0] in target and top[1] in target and top[2] in target:
                res1 += bet * r6
            else:
                res1 -= bet

        #print("sambok: %f" % (res1))
    return res1

# 3 straight win
def simulation7(pred, ans, targets=[[1,2,3],[1,2,3,4,5],[1,2,3,4,5,6]]):
    cnt = 0
    for x in targets[0]:
        for y in targets[1]:
            for z in targets[2]:
                if x != y and x != z and y != z:
                    cnt+=1
    bet = 10.0 / cnt
    i = 0
    res1 = 0
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        try:
            r7 = float(ans['samssang'][i]) - 1
        except TypeError:
            r7 = -1
        except ValueError:
            r7 = float(re.search(r'[\d.]+', ans['samssang'][i]).group()) - 1
        rcno = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        price = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno and int(ans['rank'][i]) != 1:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            price = int(ans['price'][i])
            total += 1
            i += 1
        
        price *= 30
        a = price*0.7 / r7
        r7 = (price+bet)*0.7 / (a+bet)

        if r7 * bet > 200 or r7 > 100:
            r7 *= 0.8
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()
        if total < 5 or r7 < 0:
            continue
        for x in targets[0]:
            if x > len(top):
                continue
            for y in targets[1]:
                if y > len(top):
                    continue
                for z in targets[2]:
                    if z > len(top):
                        continue
                    if x == y or x == z or y == z:
                        continue
                    if top[0] == x and top[1] == y and top[2] == z:
                        res1 += bet * r7
                    else:
                        res1 -= bet
        #print("sambok: %f" % (res1))
    return res1

if __name__ == '__main__':
    pass
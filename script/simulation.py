# -*- coding: utf-8 -*-

import parse_txt_race as pr
import datetime
import pandas as pd
import os.path
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.externals import joblib
import random
import numpy as np


# 1 win
def simulation1(pred, ans):
    #print(ans)
    i = 0
    res1 = 0
    bet = 100
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r1 = float(ans['r1'][i])
        rcno = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno and cache[int(ans['rank'][i])] == 0:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            price = int(ans['price'][i])
            total += 1
            i += 1
        a = price*0.8 / r1
        r1 = (price+300000)*0.8 / (a+300000) - 1.0
        if r1*bet > 2000:
            r1 *= 0.8
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top1 = sim_data.argmin()
        if top1 == 1:
            res1 += 100 * r1
        else:
            res1 -= 100
        #print("단승식: %f, %f" % (res1, res2))
    return res1

# 1 win in 2-3
def simulation2(pred, ans):
    i = 0
    res1 = 0
    rcno = 0
    bet = 100
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        rcno += 1
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r2 = [float(ans['r2'][i]) - 1]
        rc_no = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        while i < len(pred) and int(ans['rcno'][i]) == rc_no and cache[int(ans['rank'][i])] == 0:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            r2.append(float(ans['r2'][i]) - 1)
            total += 1
            i += 1
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top1 = sim_data.argmin()
        top = sim_data.rank()
        if total_player > 7:
            if top1 in [0, 1, 2]:
                if r2[top1] > 20:
                    r2[top1] *= 0.8
                res1 += 100 * r2[top1]
            else:
                res1 -= 100
        else:
            if top1 in [0, 1]:
                if r2[top1] > 20:
                    r2[top1] *= 0.8
                res1 += 100 * r2[top1]
            else:
                res1 -= 100

        #print("연승식: %f" % (res1))
    return res1

# 2 win
def simulation3(pred, ans):
    bet = 100
    i = 0
    res1 = 0
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r3 = float(ans['r3'][i]) - 1
        if r3*bet > 2000:
            r3 *= 0.8
        rcno = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno and cache[int(ans['rank'][i])] == 0:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            total += 1
            i += 1
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()
        if total < 2:
            continue
        if (top[0] in [1, 2]) and (top[1] in [1, 2]):
            res1 += bet * r3
        else:
            res1 -= bet

        #print("복승식: %f" % (res1))
    return res1


def get_num(line, bet):
    num_circle_list = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭"

    a = num_circle_list.find(line[:3]) / 3 + 1
    b = num_circle_list.find(line[3:6]) / 3 + 1
    r = float(line[6:]) - 1
    if r * bet > 2000:
        r *= 0.8
    return [a, b, r]


# 2 win in 3
def simulation4(pred, ans):
    bet = 100
    i = 0
    res1 = 0
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        if type(ans['bokyeon1'][i]) in [type(1), type(1.1)] or len(ans['bokyeon1'][i]) <= 6:
            return 0
        r4 = [get_num(ans['bokyeon1'][i], bet)]
        r4.append(get_num(ans['bokyeon2'][i], bet))
        r4.append(get_num(ans['bokyeon3'][i], bet))
        rcno = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno and cache[int(ans['rank'][i])] == 0:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            total += 1
            i += 1
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()
        if total < 2:
            continue
        succeed = False
        for r in r4:
            if (top[0] in [r[0], r[1]]) and (top[1] in [r[0], r[1]]):
                res1 += bet * r[2]
                succeed = True
                break
        if not succeed:
            res1 -= bet

        #print("bokyeon: %f" % (res1))
    return res1


# 2 straight win
def simulation5(pred, ans):
    bet = 100
    i = 0
    res1 = 0
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        if type(ans['ssang'][i]) in [type(1), type(1.1)] or len(ans['ssang'][i]) <= 6:
            return 0
        r5 = float(ans['ssang'][i][6:]) - 1
        if r5 * bet > 2000:
            r5 *= 0.8
        rcno = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno and cache[int(ans['rank'][i])] == 0:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            total += 1
            i += 1
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()
        if total < 2:
            continue
        if top[0] == 1 and top[1] == 2:
            res1 += bet * r5
        else:
            res1 -= bet

        #print("ssang: %f" % (res1))
    return res1


# 3 straight win
def simulation6(pred, ans):
    bet = 100
    i = 0
    res1 = 0
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        if type(ans['sambok'][i]) in [type(1), type(1.1)] or len(ans['sambok'][i]) <= 9:
            return 0
        r6 = float(ans['sambok'][i][9:]) - 1
        if r6 * bet > 2000:
            r6 *= 0.8
        rcno = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno and cache[int(ans['rank'][i])] == 0:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            total += 1
            i += 1
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()
        if total < 2:
            continue
        succeed = False
        if top[0] == 1 and top[1] == 2 and top[2] == 3:
            res1 += bet * r6
        else:
            res1 -= bet

        #print("sambok: %f" % (res1))
    return res1


def simulation_all(pred, ans):
    i = 0
    res = 0
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r1 = float(ans['r1'][i]) - 1
        r2 = [float(ans['r2'][i]) - 1]
        r3 = float(ans['r3'][i]) - 1
        rcno = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        i += 1
        total = 1
        rack_data = False
        total_player = 0
        while i < len(pred) and int(ans['rcno'][i]) == rcno and cache[int(ans['rank'][i])] == 0:
            cache[int(ans['rank'][i])] = 1
            sim_data.append(pred[i])
            r2.append(float(ans['r2'][i]) - 1)
            total_player = int(ans['cnt'][i])
            total += 1
            i += 1
        # if rack_data:
        #     continue
        sim_data = pd.Series(sim_data)
        if total < 2:
            continue
        top = sim_data.rank()
        top1 = sim_data.argmin()

        res1 = 100*r1 if top[0] == 1 else -100
        if total > 7:
            res2 = 100*r2[top1] if top[0] in [1, 2, 3] else -100
        else:
            res2 = 100*r2[top1] if top[0] in [1, 2] else -100
        res3 = 100*r3 if top[0] in [1, 2] and top[1] in [1, 2] else -100
        res += (res1 + res2 + res3)
        print("res: %f <= (%f) + (%f) + (%f)" % (res, res1, res2, res3))
    return res


if __name__ == '__main__':
    pass
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
    bet = 100
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
        while i < len(pred) and int(ans['rcno'][i]) == rcno and cache[int(ans['rank'][i])] == 0:
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
        if r1[0]*bet > 2000:
            r1[0] *= 0.8
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top1 = sim_data.argmin()

        if top1 == target:
            res1 += 100 * (r1[0] - 1)
        else:
            res1 -= 100
        #print("단승식: %f, %f" % (res1, res2))
    return res1


# 1 win in 2-3
def simulation2(pred, ans, target=0):
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
        while i < len(pred) and int(ans['rcno'][i]) == rc_no and cache[int(ans['rank'][i])] == 0:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            total_player = int(ans['cnt'][i])
            r = float(ans['r2'][i])
            a = price*0.8 / r
            r = (price+100000)*0.8 / (a+100000)
            r2.append(r-1)
            total += 1
            i += 1
        # if rack_data or total < total_player:
        #     continue

        if len(sim_data) < 6:
            continue
        sim_data = pd.Series(sim_data)
        top = int(sim_data.rank()[target])

        if total_player > 7:
            if top in [1, 2, 3]:
                if r2[top-1]*bet > 2000:
                    r2[top-1] *= 0.8
                res1 += 100 * r2[top-1]
            else:
                res1 -= 100
        else:
            if top in [1, 2]:
                if r2[top-1]*bet > 2000:
                    r2[top-1] *= 0.8
                res1 += 100 * r2[top-1]
            else:
                res1 -= 100

        #print("연승식: %f" % (res1))
    return res1

# 2 win
def simulation3(pred, ans, target=[2,3]):
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
        rcno = int(ans['rcno'][i])
        cache[int(ans['rank'][i])] = 1
        price = int(ans['price'][i])
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
        price *= 10
        a = price*0.7 / r3
        r3 = (price+100000)*0.7 / (a+100000)
        if r3*bet > 2000:
            r3 *= 0.8
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()
        if total < 2:
            continue
        if (top[0] in target) and (top[1] in target):
            res1 += bet * r3
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
        if r * bet > 2000:
            r *= 0.8
        return [a, b, r]
    except ValueError:
        return [-1, -1, -1]
    except AttributeError:
        return [-1, -1, -1]


# 2 win in 3
def simulation4(pred, ans, target=[0,1]):
    # 1,2    1,3     1,4     2,3  2,4    3,4
    #20970   -2697   23358   238 -16034  -14870

    bet = 100
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
        while i < len(pred) and int(ans['rcno'][i]) == rcno and cache[int(ans['rank'][i])] == 0:
            cache[int(ans['rank'][i])] = 1
            if ans['hr_nt'][i] == -1 or ans['jk_nt'][i] == -1 or ans['tr_nt'][i] == -1:
                rack_data = True
            sim_data.append(pred[i])
            idxes.append(ans['idx'][i])
            total_player = int(ans['cnt'][i])
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
            #print("predict: %d, %d, %d, %d, %d" % (top[0], top[1], top[2], top[3], top[4]))
            #print("predict: [%d, %d], ans: [%d, %d], r: %.0f" % (top[target[0]], top[target[1]], r[0], r[1], r[2]))
            if (idxes[int(top[target[0]])-1] in [r[0], r[1]]) and (idxes[int(top[target[1]])-1] in [r[0], r[1]]):
                res1 += bet * r[2]
                succeed = True
                break
        if not succeed:
            res1 -= bet

        #print("bokyeon: %f" % (res1))
    return res1


# 2 straight win
def simulation5(pred, ans, targets=[[2,3],[3,2]]):
    bet = 100 / len(targets)
    i = 0
    res1 = 0
    assert len(pred) == len(ans)
    while True:
        cache = np.zeros(20)
        if i >= len(pred):
            break
        sim_data = [pred[i]]
        r5 = float(ans['ssang'][i]) - 1
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
def simulation6(pred, ans, targets=[[0,1,3], [0,1,4], [0,2,3], [0,2,4], [0,3,4]]):
    bet = 100 / len(targets)
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
        while i < len(pred) and int(ans['rcno'][i]) == rcno and cache[int(ans['rank'][i])] == 0:
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

        if r6 * bet > 2000:
            r6 *= 0.8
        # if rack_data or total < total_player:
        #     continue
        sim_data = pd.Series(sim_data)
        top = sim_data.rank()
        if total < 5 or r6 == -1:
            continue
        
        for target in targets:
            if top[target[0]] in [1, 2, 3] and top[target[1]] in [1, 2, 3] and top[target[2]] in [1, 2, 3]:
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
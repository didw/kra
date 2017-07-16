# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os.path
import glob
from sklearn.externals import joblib

DEBUG = False
def make_one_hot(name):
    trainer_list = ["강대은", "강영진", "고성동", "고영덕", "김성오", "김신호", "김영래", "김영복", "김태준", "김한철", "박병진", "백인호", "변용호", "신경호", "심도연", "윤덕상", "이준호", "이태용", "임용찬", "장성종", "정성훈", "정영수", "좌윤철", "최기호", "한상배"]
    res = np.zeros(len(trainer_list))
    for i in range(len(trainer_list)):
        if name == trainer_list[i]:
            res[i] = 1
    return res

def get_trainer(name):
    return make_one_hot(name)


if __name__ == '__main__':
    print(len(get_trainer("이인호")))


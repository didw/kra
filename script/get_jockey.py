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
    jockey_list = ["강성", "강수한", "강영진", "곽용남", "권동석", "권육", "권진영", "김경훈", "김경휴", "김기섭", "김다영", "김대연", "김명호", "김영수", "김용섭", "김이랑", "김정일", "김주희", "김준호", "김태준", "김한남", "김형준", "김홍권", "나유나", "문성호", "문현진", "박기영", "박성광", "박정민", "박준호", "박훈", "배성아", "심광선", "심도연", "심태섭", "안득수", "원유일", "유미라", "윤도선", "이덕형", "이동준", "이장우", "이재웅", "이태용", "이현섭", "임재광", "장우성", "전현준", "정명일", "정성훈", "정영수", "조희원", "한영민", "허회창", "황태선"]
    res = np.zeros(len(jockey_list))
    for i in range(len(jockey_list)):
        if name == jockey_list[i]:
            res[i] = 1
    return res

def get_jockey(name):
    return make_one_hot(name)


if __name__ == '__main__':
    print(len(get_jockey("아담")))


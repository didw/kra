# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os.path
import glob
from sklearn.externals import joblib
from operator import itemgetter

class _RegExLib:
    """Set up regular expressions"""
    # use https://regexper.com to visualise these if required
    _reg_title = re.compile(r'제목 +:')
    _reg_rcno = re.compile(r'(?<=제 *)\d(?=경주)')
    _reg_region = re.compile(r'\(서울\)')
    _reg_course = re.compile(r'\d{4}(?=M)')
    _reg_rc_grade = re.compile(r'(?<=M *)[가-힣\d]+(?= )')
    _reg_rc_type = re.compile(r'(?<= )[가-힣A-Z]+(?=경주명)')
    _reg_rc_name = re.compile(r'(?<=경주명 *: *)[가-힣]+')
    _reg_rc_cond = re.compile(r'(?<=경주조건\s*:\s+)\S+')
    _reg_rc_age = re.compile(r'(?<=경주조건\s*:\s+\S+\s+)\S+')
    _reg_rc_weather = re.compile(r'(?<=날씨:)\S+')
    _reg_course_cond = re.compile(r'(?<=주로:)\S+')
    _reg_humidity = re.compile(r'(?<=\()\d+(?=\%)')
    _reg_win_prize = re.compile(r'(?<=착순상금:\s+).*')
    _reg_add_prize = re.compile(r'(?<=부가상금:\s+).*')
    _reg_dash = re.compile(r'[─-]+')

    _reg_get_info01 = re.compile(r'부담중량')
    _reg_rank = re.compile(r'\s*\d+')
    _reg_no_gate = re.compile(r'(?<=\s*\d+\s+)\d+')
    _reg_hr_name = re.compile(r'(?<=\s*\d+\s+\d+\s+)\S+(?=\s*\S[\(포\)]*\s+[암수거])')
    _reg_hr_hometown = re.compile(r'(?<=\s*\d+\s+\d+\s+\S+\s*)\S[\(포\)]*(?=\s+[암수거])')
    _reg_hr_gender = re.compile(r'(?<=\s*\d+\s+\d+\s+\S+\s*\S[\(포\)]*\s+)[암수거]')
    _reg_hr_age = re.compile(r'(?<=\s*\d+\s+\d+\s+\S+\s*\S[\(포\)]*\s+[암수거]\s+)\d')
    _reg_hr_impost = re.compile(r'\d+\.\d')
    _reg_jk_name = re.compile(r'(?<=\d+\.\d\s+)[가-힣]+')
    _reg_tr_name = re.compile(r'(?<=\d+\.\d\s+[가-힣]+\s+)[가-힣]+')
    _reg_owner_name = re.compile(r'(?<=\d+\.\d\s+[가-힣]+\s+[가-힣]+\s+)[가-힣\s]+(?=\d+)')
    _reg_rating = re.compile(r'(?<=\d+\.\d\s+[가-힣]+\s+[가-힣]+\s+[가-힣\s]+)[\d ]+')

    _reg_get_info02 = re.compile(r'마\s*체\s*중')
    _reg_weight = re.compile(r'\d+(?=\()')
    _reg_dweight = re.compile(r'(?<=\([ +-]*)\d+(?=\))')
    _reg_rc_score = re.compile(r'(?<=\)\s+)\d+:\d+\.\d+')
    _reg_dist = re.compile(r'(?<=\d+:\d+\.\d+\s+)\S+')
    _reg_score_detail = re.compile(r'[\d\s]+-[\d\s]+-[\d\s]+-[\d\s]+-[\d\s]+-[\d\s]+')

    _reg_get_info03 = re.compile(r'1F')
    _reg_g3f = re.compile(r'(?<=\s+\d+\s+)\d+\.\d')
    _reg_s1f = re.compile(r'(?<=\s+\d+\.\d\s+)\d:\d+\.\d')
    _reg_1c = re.compile(r'(?<=\s+\d+\.\d\s+\d:\d+\.\d\s+)\d:\d+\.\d')
    _reg_2c = re.compile(r'(?<=\s+\d+\.\d\s+\d:\d+\.\d\s+\d:\d+\.\d\s+)\d:\d+\.\d')
    _reg_3c = re.compile(r'\d:\d+\.\d(?=\s+\d:\d+\.\d\s+\d+\.\d)')
    _reg_4c = re.compile(r'\d:\d+\.\d(?=\s+\d+\.\d)')
    _reg_g1f = re.compile(r'\d+\.\d(?=\s+\d+\.\d\s+\d+\.\d)')
    _reg_bet_win = re.compile(r'(?<=\s+\d+\.\d\s+)\d+\.\d(?=\s+\d+\.\d)')
    _reg_bet_perfecta = re.compile(r'(?<=\s+\d+\.\d\s+\d+\.\d\s+)\d+\.\d')

    _reg_total_sales = re.compile(r'매출액')
    _reg_total_sales_dan = re.compile(r'(?<=단식:\s+)[\d,]+')
    _reg_total_sales_yeon = re.compile(r'(?<=연식:\s+)[\d,]+')
    _reg_total_sales_bok = re.compile(r'(?<=복식:\s+)[\d,]+')
    _reg_total_sales_bokyeon = re.compile(r'(?<=복연:\s+)[\d,]+')
    _reg_total_sales_ssang = re.compile(r'(?<=쌍식:\s+)[\d,]+')
    _reg_total_sales_sambok = re.compile(r'(?<=삼복:\s+)[\d,]+')
    _reg_total_sales_samssang = re.compile(r'(?<=삼쌍:\s+)[\d,]+')
    _reg_total_sales_total = re.compile(r'(?<=합계:\s+)[\d,]+')

    _reg_bet_rate = re.compile(r'배당률\s')
    _reg_bet_rate_dan = re.compile(r'(?<=단:\s+)\S\d+\.\d')
    _reg_bet_rate_yeon = re.compile(r'(?<=연:\s+)\S\d+\.\d\s+\S\d+\.\d\s+\S\d+\.\d')
    _reg_bet_rate_bok = re.compile(r'(?<=복:\s+)\S{2}\d+\.\d')
    _reg_bet_rate_ssang = re.compile(r'(?<=쌍:\s+)\S{2}\d+\.\d')
    _reg_bet_rate_bokyeon = re.compile(r'(?<=복연:\s+)\S{2}\d+\.\d\s+\S{2}\d+\.\d\s+\S{2}\d+\.\d')
    _reg_bet_rate_sambok = re.compile(r'(?<=삼복:\s+)\S{3}\d+\.\d')
    _reg_bet_rate_samssang = re.compile(r'(?<=삼쌍:\s+)\S{3}\d+\.\d')

    _reg_FL = re.compile(r'(?<=펄\s*롱:\s*).*')
    _reg_passM = re.compile(r'(?<=통과\s*M:\s*).*')
    _reg_passT = re.compile(r'(?<=통과\s*T:\s*).*')

    _reg_rank_s1f = re.compile(r'(?<=통과순위\s*S[-]1F\s*:\s*).*')
    _reg_rank_c1 = re.compile(r'(?<=C1\s*:\s*).*')
    _reg_rank_c2 = re.compile(r'(?<=C2\s*:\s*).*')
    _reg_rank_c3 = re.compile(r'(?<=C3\s*:\s*).*')
    _reg_rank_c4 = re.compile(r'(?<=C4\s*:\s*).*')
    _reg_rank_g1f = re.compile(r'(?<=G[-]1F\s*:\s*).*')

    _reg_etc = re.compile(r'세부내용')
    _reg_etc_type = re.compile(r'(기수|말)')
    _reg_etc_no = re.compile(r'(?<=(기수|말)\s+)\d+')
    _reg_etc_name = re.compile(r'(?<=(기수|말)\s+\d+\s+)\S+')
    _reg_etc_kind = re.compile(r'(?<=(기수|말)\s+\d+\s+\S+\s+)\S+')
    _reg_etc_not_content = re.compile(r'\s{25}')  # check content is blank
    _reg_etc_content = re.compile(r'(?<=(기수|말)\s+\d+\s+\S+\s+\S+\s+)\S+')
    _reg_etc_detail01 = re.compile(r'(?<=(기수|말)\s+\d+\s+\S+\s+\S+\s+).+')  # if content is blank
    _reg_etc_detail02 = re.compile(r'(?<=(기수|말)\s+\d+\s+\S+\s+\S+\s+\S+\s+).+')  # if content is not blank


    __slots__ = ['title','rcno','region','course','rc_grade','rc_type','rc_name',
                'rc_cond','rc_age','rc_weather','course_cond','humidity','win_prize',
                'add_prize','dash', 'get_info01','rank','no_gate','hr_name','hr_hometown',
                'hr_gender','hr_age','hr_impost','jk_name','tr_name','owner_name',
                'rating','get_info02','weight','dweight','rc_score','dist','score_detail',
                'get_info03','g3f','s1f','get_1c','get_2c','get_3c','get_4c','g1f','bet_win','bet_perfecta',
                'total_sales','total_sales_dan','total_sales_yeon','total_sales_bok',
                'total_sales_bokyeon','total_sales_ssang','total_sales_sambok','total_sales_samssang',
                'total_sales_total','bet_rate','bet_rate_dan','bet_rate_yeon','bet_rate_bok',
                'bet_rate_ssang','bet_rate_bokyeon','bet_rate_sambok','bet_rate_samssang',
                'FL','passM','passT','rank_s1f','rank_c1','rank_c2','rank_c3','rank_c4',
                'rank_g1f','etc','etc_type','etc_no','etc_name','etc_kind','etc_not_content',
                'etc_content','etc_detail01','etc_detail02']

    def __init__(self, line):
        # check whether line has a positive match with all of the regular expressions
        self.title = self._reg_title.match(line)
        self.rcno = self._reg_rcno.search(line)
        self.region = self._reg_region.search(line)
        self.course = self._reg_course.search(line)
        self.rc_grade = self._reg_rc_grade.search(line)
        self.rc_type = self._reg_rc_type.search(line)
        self.rc_name = self._reg_rc_name.search(line)
        self.rc_cond = self._reg_rc_cond.search(line)
        self.rc_age = self._reg_rc_age.search(line)
        self.rc_weather = self._reg_rc_weather.search(line)
        self.course_cond = self._reg_course_cond.search(line)
        self.humidity = self._reg_humidity.search(line)
        self.win_prize = self._reg_win_prize.search(line)
        self.add_prize = self._reg_add_prize.search(line)
        self.dash = self._reg_dash.search(line)

        self.get_info01 = self._reg_get_info01.search(line)
        self.rank = self._reg_rank.search(line)
        self.no_gate = self._reg_no_gate.search(line)
        self.hr_name = self._reg_hr_name.search(line)
        self.hr_hometown = self._reg_hr_hometown.search(line)
        self.hr_gender = self._reg_hr_gender.search(line)
        self.hr_age = self._reg_hr_age.search(line)
        self.hr_impost = self._reg_hr_impost.search(line)
        self.jk_name = self._reg_jk_name.search(line)
        self.tr_name = self._reg_tr_name.search(line)
        self.owner_name = self._reg_owner_name.search(line)
        self.rating = self._reg_rating.search(line)

        self.get_info02 = self._reg_get_info02.search(line)
        self.weight = self._reg_weight.search(line)
        self.dweight = self._reg_dweight.search(line)
        self.rc_score = self._reg_rc_score.search(line)
        self.dist = self._reg_dist.search(line)
        self.score_detail = self._reg_score_detail.search(line)

        self.get_info03 = self._reg_get_info03.search(line)
        self.g3f = self._reg_g3f.search(line)
        self.s1f = self._reg_s1f.search(line)
        self.get_1c = self._reg_1c.search(line)
        self.get_2c = self._reg_2c.search(line)
        self.get_3c = self._reg_3c.search(line)
        self.get_4c = self._reg_4c.search(line)
        self.g1f = self._reg_g1f.search(line)
        self.bet_win = self._reg_bet_win.search(line)
        self.bet_perfecta = self._reg_bet_perfecta.search(line)

        self.total_sales = self._reg_total_sales.search(line)
        self.total_sales_dan = self._reg_total_sales_dan.search(line)
        self.total_sales_yeon = self._reg_total_sales_yeon.search(line)
        self.total_sales_bok = self._reg_total_sales_bok.search(line)
        self.total_sales_bokyeon = self._reg_total_sales_bokyeon.search(line)
        self.total_sales_ssang = self._reg_total_sales_ssang.search(line)
        self.total_sales_sambok = self._reg_total_sales_sambok.search(line)
        self.total_sales_samssang = self._reg_total_sales_samssang.search(line)
        self.total_sales_total = self._reg_total_sales_total.search(line)

        self.bet_rate = self._reg_bet_rate.search(line)
        self.bet_rate_dan = self._reg_bet_rate_dan.search(line)
        self.bet_rate_yeon = self._reg_bet_rate_yeon.search(line)
        self.bet_rate_bok = self._reg_bet_rate_bok.search(line)
        self.bet_rate_ssang = self._reg_bet_rate_ssang.search(line)
        self.bet_rate_bokyeon = self._reg_bet_rate_bokyeon.search(line)
        self.bet_rate_sambok = self._reg_bet_rate_sambok.search(line)
        self.bet_rate_samssang = self._reg_bet_rate_samssang.search(line)

        self.FL = self._reg_FL.search(line)
        self.passM = self._reg_passM.search(line)
        self.passT = self._reg_passT.search(line)

        self.rank_s1f = self._reg_rank_s1f.search(line)
        self.rank_c1 = self._reg_rank_c1.search(line)
        self.rank_c2 = self._reg_rank_c2.search(line)
        self.rank_c3 = self._reg_rank_c3.search(line)
        self.rank_c4 = self._reg_rank_c4.search(line)
        self.rank_g1f = self._reg_rank_g1f.search(line)

        self.etc = self._reg_etc.search(line)
        self.etc_type = self._reg_etc_type.search(line)
        self.etc_no = self._reg_etc_no.search(line)
        self.etc_name = self._reg_etc_name.search(line)
        self.etc_kind = self._reg_etc_kind.search(line)
        self.etc_not_content = self._reg_etc_not_content.search(line)
        self.etc_content = self._reg_etc_content.search(line)
        self.etc_detail01 = self._reg_etc_detail01.search(line)
        self.etc_detail02 = self._reg_etc_detail02.search(line)


def parse_race_detail(filename):
    posts = {}
    posts["rc_date"]=os.path.basename(filename)[-12:-4]
    in_file = open(filename)
    line = next(in_file)
    data = dict()
    while True:
        #line = unicode(line, 'euc-kr').encode('utf-8')
        if line is None or len(line) == 0:
            break
        # 제목 : 16년12월25일(일)  제15경주
        date = int(re.search(r'\d{8}', filename).group())
        if re.search(r'\(서울\)', line) is not None:
            course = int(re.search(r'\d+00*(?=M)', line).group())
        if re.search(r'경주조건', line) is not None and re.search(r'주로', line) is not None:
            try:
                humidity = int(re.search(r'(?<=\()[\d ]+(?=\%\))', line).group())
            except:
                humidity = 10
        # read name
        if re.search(r'부담중량', line) is not None and re.search(r'산지', line) is not None:
            name_list = []
            while re.search(r'[-─]{5}', line) is None:
                line = in_data.readline()
                break
            while True:
                line = in_data.readline()
                #line = unicode(line, 'euc-kr').encode('utf-8')
                if re.search(r'[-─]{5}', line) is not None:
                    res = re.search(r'[-─]{5}', line).group()
                    if DEBUG: print("break: %s" % res)
                    break
                try:
                    name = re.findall(r'\S+', line)[2].replace('★', '')
                except:
                    print("except: in %s : %s" % (filename, line))
                name_list.append(name)
                data[name] = [date, course, humidity]
                if DEBUG:
                    print("name: %s" % unicode(name, 'utf-8'))

        # read score
        if re.search(r'단승식', line) is not None:
            while re.search(r'[-─]{10}', line) is None:
                line = in_data.readline()
                break
            i = 0
            while True:
                line = in_data.readline()
                #line = unicode(line, 'euc-kr').encode('utf-8')
                if re.search(r'[-─]{10}', line) is not None:
                    res = re.search(r'[-─]{10}', line).group()
                    #print("result: %s" % res)
                    break
                words = re.findall(r'\S+', line)
                s1f, g1f, g2f, g3f = -1, -1, -1, -1
                if len(words) == 9:
                    if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (words[3], words[6], words[2]))
                    try:
                        g1f = float(re.search(r'\d{2}\.\d', words[6]).group())*10
                    except:
                        print("parsing error in race_detail - 1")
                        g1f = -1
                elif len(words) == 11:
                    if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (words[3], words[8], words[2]))
                    try:
                        g1f = float(re.search(r'\d{2}\.\d', words[8]).group())*10
                    except:
                        print("parsing error in race_detail - 3")
                        g1f = -1
                elif len(words) < 9:
                    #print("unexpected line: %s" % line)
                    continue
                try:
                    s1f = float(re.search(r'\d{2}\.\d', words[3]).group())*10
                except:
                    s1f = 150
                try:
                    g3f = float(re.search(r'\d{2}\.\d', words[2]).group())*10
                except:
                    g3f = 400
                if s1f < 100 or s1f > 200:
                    s1f = -1
                if g1f < 100 or g1f > 200:
                    g1f = -1
                if g3f < 300 or g3f > 500:
                    g3f = -1
                data[name_list[i]].extend([s1f, g1f, g3f])
                i += 1
    line = next(in_file)


def parse_ap_rslt(filename):
    in_data = open(filename)
    data = dict()
    course = 900
    while True:
        line = in_data.readline()
        #line = line.encode('utf-8')

        if line is None or len(line) == 0:
            break
        # 제목 : 16년12월25일(일)  제15경주
        date = int(re.search(r'\d{8}', filename).group())
        if re.search(r'주로상태', line) is not None and re.search(r'날.+씨', line) is not None:
            try:
                humidity = int(re.search(r'(?<=\()[\d ]+(?=\%\))', line).group())
            except:
                humidity = 10
        # read name
        if re.search(r'산지', line) is not None and (re.search(r'선수명', line) is not None or re.search(r'기수명', line) is not None):
            name_list = []
            while re.search(r'[-─]{5}', line) is None:
                line = in_data.readline()
                break
            while True:
                line = in_data.readline()
                #line = unicode(line, 'euc-kr').encode('utf-8')
                if re.search(r'[-─]{5}', line) is not None:
                    res = re.search(r'[-─]{5}', line).group()
                    if DEBUG: print("break: %s" % res)
                    break
                try:
                    name = re.findall(r'\S+', line)[2].replace('★', '')
                except:
                    print("except: in %s : %s" % (filename, line))
                name_list.append(name)
                data[name] = [date, course, humidity]
                if DEBUG:
                    print("name: %s" % unicode(name, 'utf-8'))

        # read score
        if re.search(r'S-1F', line) is not None:
            while re.search(r'[-─]{10}', line) is None:
                line = in_data.readline()
                break
            i = 0
            while True:
                line = in_data.readline()
                #line = unicode(line, 'euc-kr').encode('utf-8')
                if re.search(r'[-─]{10}', line) is not None:
                    res = re.search(r'[-─]{10}', line).group()
                    if DEBUG: print("break: %s" % res)
                    break
                words = re.findall(r'\S+', line)
                if len(words) < 9:
                    i += 1
                    continue
                if DEBUG: print("s1f: %s, g1f: %s, g3f: %s" % (words[3], words[6], words[2]))
                try:
                    s1f = float(re.search(r'\d{2}\.\d', words[3]).group())*10
                except:
                    print("parsing error in ap s1f")
                    s1f = -1
                try:
                    g1f = float(re.search(r'\d{2}\.\d', words[6]).group())*10
                except:
                    print("parsing error in ap g1f")
                    g1f = -1
                try:
                    g3f = float(re.search(r'\d{2}\.\d', words[2]).group())*10
                except:
                    print("parsing error in ap g3f")
                    g3f = -1
                if s1f < 100 or s1f > 200:
                    s1f = -1
                if g1f < 100 or g1f > 200:
                    g1f = -1
                if g3f < 300 or g3f > 500:
                    g3f = -1
                data[name_list[i]].extend([s1f, g1f, g3f])
                i += 1
    for k,v in data.items():#.iteritems():
        if len(v) < 5:
            continue
        if k in data:
            data[k].append(v)
        else:
            data[k] = [v]


if __name__ == '__main__':
    DEBUG = True
    fname1 = '../txt/1/ap-check-rslt/ap-check-rslt_1_20070622.txt'
    fname2 = '../txt/1/rcresult/rcresult_1_20091213.txt'
    parse_ap_rslt(fname1)
    parse_race_detail(fname2)


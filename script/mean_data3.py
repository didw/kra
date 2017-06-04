# -*- coding:utf-8 -*-


from __future__ import print_function
try:
    # For Python 3.0 and later
    from urllib.request import urlopen, Request
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen, Request
import os
from bs4 import BeautifulSoup
import datetime
import re
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from etaprogress.progress import ProgressBar

DEBUG = False

def get_csv():
    df = pd.read_csv('../data/1_2007_2016_v1.csv', index_col='date')
    print(np.shape(df))
    m_c = {}
    for c in df['course'].unique():
        m = df.loc[df['course']==c, 'rctime'].mean()
        df = df.loc[(df['course']!=c) | (df['rctime'] < 1.05*m)] # remove outlier
        print(c, m)
    print(np.shape(df))
    return df


class cmake_mean:
    def __init__(self):
        self.df = get_csv()

    def get_data(self):
        self.data = {}
        name_list = self.df['name'].unique()

        self.data = {}
        self.data['course'] = {}
        for c in name_list:
            self.data[c] = {'humidity':{}, 'month':{}, 'age':{}, 'hr_days':{}, 'lastday':{}, 'idx':{}, 'rcno':{}, 'budam':{}, 'dbudam':{}, 'jockey':{}, 'trainer':{}, 'rank':{}}
            for j in range(1,82):
                self.data[c]['jc%d'%j] = {}

        self.mean_data = {'course':{}, 'humidity':{}, 'month':{}, 'age':{}, 'hr_days':{}, 'lastday':{}, 'idx':{}, 'rcno':{}, 'budam':{}, 'dbudam':{}, 'jockey':{}, 'trainer':{}, 'rank':{}}
        for j in range(1,82):
            self.mean_data['jc%d'%j] = {}

        def add_course(row):
            if row['course'] not in self.data['course'].keys():
                self.data['course'][row['course']] = [row['rctime']]
            else:
                self.data['course'][row['course']].append(row['rctime'])

        def add_item(row, item):
            if row[item] not in self.data[row['name']][item].keys():
                self.data[row['name']][item][row[item]] = [row['rctime']/self.mean_data['course'][row['course']]]
            else:
                self.data[row['name']][item][row[item]].append(row['rctime']/self.mean_data['course'][row['course']])

        def add_item_days(row, item, day):
            if int(row[item]/day)*day not in self.data[row['name']][item].keys():
                self.data[row['name']][item][int(row[item]/day)*day] = [row['rctime']/self.mean_data['course'][row['course']]]
            else:
                self.data[row['name']][item][int(row[item]/day)*day].append(row['rctime']/self.mean_data['course'][row['course']])

        def make_norm(c, item):
            average = []
            for key in self.df[item].unique():
                try:
                    self.data[c][item][key] = np.mean(self.data[c][item][key])
                    average.append(self.data[c][item][key])
                except KeyError:
                    self.data[c][item][key] = 1
            average = np.mean(average)
            for key in self.df[item].unique():
                if self.data[c][item][key] != 1:
                    self.data[c][item][key] = self.data[c][item][key] / average

        def make_norm_day(c, item, day):
            average = []
            max_value = 4000 if day == 100 else 800
            for key in range(0, max_value, day):
                try:
                    self.data[c][item][key] = np.mean(self.data[c][item][key])
                    average.append(self.data[c][item][key])
                except KeyError:
                    self.data[c][item][key] = 1
            average = np.mean(average)
            for key in range(0, max_value, day):
                if self.data[c][item][key] != 1:
                    self.data[c][item][key] = self.data[c][item][key] / average

        def make_mean(item_class):
            bar = ProgressBar(len(self.df[item_class].unique())*len(name_list), max_width=80)
            print("\nprocessing Data [%s]"%item_class)
            item_list = {}
            for item in self.df[item_class].unique():
                item_list[item] = []
                for c in name_list:
                    bar.numerator += 1
                    if bar.numerator%100 == 0:
                        print("%s" % (bar,), end='\r')
                    if self.data[c][item_class][item] == 1.0:
                        continue
                    try:
                        item_list[item].append(self.data[c][item_class][item])
                    except KeyError:
                        print("Key Error: ", c, item_class, item)
                try:
                    if len(item_list[item]) == 0:
                        self.mean_data[item_class][item] = 1.0
                    else:
                        self.mean_data[item_class][item] = np.mean(item_list[item])
                except ValueError:
                    print("ValueError: ", item_class, item)
                except TypeError:
                    print("TypeError: ", item_class, item)

        def make_mean_day(item_class, day):
            max_value = 4000 if day == 100 else 800
            bar = ProgressBar(int(max_value/day)*len(name_list), max_width=80)
            print("\nprocessing Data [%s]"%item_class)
            item_list = {}
            for key in range(0, max_value, day):
                item_list[key] = []
                for c in name_list:
                    bar.numerator += 1
                    if bar.numerator%100 == 0:
                        print("%s" % (bar,), end='\r')
                    if self.data[c][item_class][key] == 1.0:
                        continue
                    try:
                        item_list[key].append(self.data[c][item_class][key])
                    except KeyError:
                        print("Key Error: ", c, item_class, key)
                try:
                    self.mean_data[item_class][key] = np.mean(item_list[key])
                except ValueError:
                    print("ValueError: ", item_class, key)
                except TypeError:
                    print("TypeError: ", item_class, key)

        print("Loading data1")
        bar = ProgressBar(len(self.df), max_width=80)
        for i in range(len(self.df)):
            bar.numerator += 1
            if bar.numerator%100 == 0:
                print("%s" % (bar,), end='\r')
            row = self.df.iloc[i,:]
            add_course(row)
        for key in self.df['course'].unique():
            self.mean_data['course'][key] = np.mean(self.data['course'][key])

        print("\n\nLoading data2")
        bar = ProgressBar(len(self.df), max_width=80)
        for i in range(len(self.df)):
            bar.numerator += 1
            if bar.numerator%100 == 0:
                print("%s" % (bar,), end='\r')
            row = self.df.iloc[i,:]

            add_item(row, 'humidity')
            add_item(row, 'month')
            add_item(row, 'age')
            add_item_days(row, 'hr_days', 100)
            add_item_days(row, 'lastday', 10)
            add_item(row, 'idx')
            add_item(row, 'rcno')
            add_item(row, 'budam')
            add_item(row, 'dbudam')
            add_item(row, 'jockey')
            add_item(row, 'trainer')
            add_item(row, 'rank')
            for j in range(1,82):
                add_item(row, 'jc%d'%j)

        print("\n\nNormalize data")
        bar = ProgressBar(len(name_list), max_width=80)
        for c in name_list:
            bar.numerator += 1
            print("%s" % (bar,), end='\r')
            make_norm(c, 'humidity')
            make_norm(c, 'month')
            make_norm(c, 'age')
            make_norm_day(c, 'hr_days', 100)
            make_norm_day(c, 'lastday', 10)
            make_norm(c, 'idx')
            make_norm(c, 'rcno')
            make_norm(c, 'budam')
            make_norm(c, 'dbudam')
            make_norm(c, 'jockey')
            make_norm(c, 'trainer')
            make_norm(c, 'rank')
            for j in range(1, 82):
                make_norm(c, 'jc%d'%j)

        make_mean('humidity')
        make_mean('month')
        make_mean('age')
        make_mean_day('hr_days', 100)
        make_mean_day('lastday', 10)
        make_mean('idx')
        make_mean('rcno')
        make_mean('budam')
        make_mean('dbudam')
        make_mean('jockey')
        make_mean('trainer')
        make_mean('rank')
        for j in range(1,82):
            make_mean('jc%d'%j)

        print(self.mean_data)


class mean_data2:
    def __init__(self):
        self.mean_pkl = joblib.load('../data/1_2007_2016_v1_md3.pkl')
        md = self.mean_pkl.mean_data
        fout = open('../data/1_2007_2016_v1_md3.csv', 'wt')
        fout.write("write to csv\n")
        for k1 in md.keys():
            fout.write("%s\n"%k1)
            for k2 in md[k1].keys():
                fout.write("%s\t%f\n"%(k2, md[k1][k2]))
        fout.close()

    def make_humidity(self):
        pass

    def make_data(self):
        pass


if __name__ == '__main__':
    DEBUG = True
    m = cmake_mean()
    m.get_data()
    joblib.dump(m, '../data/1_2007_2016_v1_md3.pkl')
    md = mean_data2()


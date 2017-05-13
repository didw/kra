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


class Make_mean:
    def __init__(self):
        self.df = get_csv()

    def get_humidity(self):
        self.humidity = {}
        humidity_list = self.df['humidity'].unique()
        course_list = self.df['course'].unique()
        hourse_list = self.df['name'].unique()
        for c in course_list:
            for horse in hourse_list:
                m = {}
                mean_list = []
                for item in humidity_list:
                    selected = self.df.loc[(self.df['name']==horse) & (self.df['course']==c) & (self.df['humidity']==item), 'rctime']
                    if len(selected) != 0:
                        m[item] = selected.mean()
                        mean_list.append(selected.mean())
                    else:
                        m[item] = 1
                total_average = np.mean(mean_list)
                for item in humidity_list:
                    if m[item] != 1:
                        m[item] = m[item] / total_average
                    try:
                        self.humidity[item].append(m[item])
                    except KeyError:
                        self.humidity[item] = [m[item]]
        for item in humidity_list:
            self.humidity[item] = np.mean(self.humidity[item])
        print(self.humidity)

    def get_month(self):
        self.month = {}
        humidity_list = self.df['month'].unique()
        course_list = self.df['course'].unique()
        horse_list = self.df['name'].unique()
        for c in course_list:
            for horse in horse_list:
                m = {}
                mean_list = []
                for item in humidity_list:
                    selected = self.df.loc[(self.df['name']==horse) & (self.df['course']==c) & (self.df['month']==item), 'rctime']
                    if len(selected) != 0:
                        m[item] = selected.mean()
                        mean_list.append(selected.mean())
                    else:
                        m[item] = 1
                total_average = np.mean(mean_list)
                for item in humidity_list:
                    if m[item] != 1:
                        m[item] = m[item] / total_average
                    try:
                        self.month[item].append(m[item])
                    except KeyError:
                        self.month[item] = [m[item]]
        for item in humidity_list:
            self.month[item] = np.mean(self.month[item])
        print(self.month)

    def get_data(self):
        self.data = {}
        print("Loading data")
        course_list = self.df['course'].unique()
        bar = ProgressBar(len(self.df), max_width=80)
        for i in range(len(self.df)):
            bar.numerator += 1
            if bar.numerator%100 == 0:
                print("%s" % (bar,), end='\r')
            row = self.df.iloc[i,:]

            self.data = {}
            for c in course_list:
                self.data[c] = {'course':{}, 'humidity':{}, 'month':{}, 'age':{}, 'hr_days':{}, 'lastday':{}, 'idx':{}, 'rcno':{}, 'budam':{}, 'dbudam':{}, 'jockey':{}, 'trainer':{}}
                for j in range(1,82):
                    self.data[c]['jc%d'%j] = {}

            def add_item(data, row, item):
                try:
                    data[row['course']][item][row[item]].append(row['rctime'])
                except KeyError:
                    data[row['course']][item][row[item]] = [row['rctime']]

            def add_item_days(data, row, item, day):
                try:
                    data[row['course']][item][int(row[item]/day)*day].append(row['rctime'])
                except KeyError:
                    data[row['course']][item][int(row[item]/day)*day] = [row['rctime']]

            add_item(self.data, row, 'course')
            add_item(self.data, row, 'humidity')
            add_item(self.data, row, 'month')
            add_item(self.data, row, 'age')
            add_item_days(self.data, row, 'hr_days', 30)
            add_item_days(self.data, row, 'lastday', 10)
            add_item(self.data, row, 'idx')
            add_item(self.data, row, 'rcno')
            add_item(self.data, row, 'budam')
            add_item(self.data, row, 'dbudam')
            add_item(self.data, row, 'jockey')
            add_item(self.data, row, 'trainer')
            for j in range(1,82):
                add_item(self.data, row, 'jc%d'%j)

        def make_norm(data, name, c, item):
            average = []
            for key in self.df[item].unique():
                try:
                    data[name][c][item][key] = np.mean(data[name][c][item][key])
                    average.append(data[name][c][item][key])
                except KeyError:
                    data[name][c][item][key] = 1
            average = np.mean(average)
            for key in self.df[item].unique():
                if data[name][c][item][key] != 1:
                    data[name][c][item][key] = data[name][c][item][key] / average

        def make_norm_day(data, name, c, item, day):
            average = []
            max_value = 4000 if day == 30 else 800
            for key in range(0, max_value, day):
                try:
                    data[name][c][item][key] = np.mean(data[name][c][item][key])
                    average.append(data[name][c][item][key])
                except KeyError:
                    data[name][c][item][key] = 1
            average = np.mean(average)
            for key in range(0, max_value, day):
                if data[name][c][item][key] != 1:
                    data[name][c][item][key] = data[name][c][item][key] / average

        print("\n\nNormalize data")
        bar = ProgressBar(len(course_list), max_width=80)
        for c in course_list:
            bar.numerator += 1
            print("%s" % (bar,), end='\r')
            make_norm(self.data, name, c, 'humidity')
            make_norm(self.data, name, c, 'month')
            make_norm(self.data, name, c, 'age')
            make_norm_day(self.data, name, c, 'hr_days', 30)
            make_norm_day(self.data, name, c, 'lastday', 10)
            make_norm(self.data, name, c, 'idx')
            make_norm(self.data, name, c, 'rcno')
            make_norm(self.data, name, c, 'budam')
            make_norm(self.data, name, c, 'dbudam')
            make_norm(self.data, name, c, 'jockey')
            make_norm(self.data, name, c, 'trainer')
        for j in range(1, 82):
            make_norm(self.data, name, c, 'jc%d'%j)

        self.mean_data = {'humidity':{}, 'month':{}, 'age':{}, 'hr_days':{}, 'lastday':{}, 'idx':{}, 'rcno':{}, 'budam':{}, 'dbudam':{}, 'jockey':{}, 'trainer':{}}
        for j in range(1,82):
            self.mean_data['jc%d'%j] = {}

        def make_mean(item_class):
            bar = ProgressBar(len(self.df[item_class].unique())*len(course_list), max_width=80)
            print("\nprocessing Data [%s]"%item_class)
            item_list = {}
            for item in self.df[item_class].unique():
                item_list[item] = []
                for name in self.df['name'].unique():
                    for c in course_list:
                        bar.numerator += 1
                        if bar.numerator%100 == 0:
                            print("%s" % (bar,), end='\r')
                        try:
                            item_list[item].append(self.data[name][c][item_class][item])
                        except KeyError:
                            print("Key Error: ", name, c, item_class, item)
                try:
                    self.mean_data[item_class][item] = np.mean(item_list[item])
                except ValueError:
                    print("ValueError: ", item_class, item)
                except TypeError:
                    print("TypeError: ", item_class, item)

        def make_mean_day(item_class, day):
            max_value = 4000 if day == 30 else 800
            bar = ProgressBar(int(max_value/day)*len(course_list), max_width=80)
            print("\nprocessing Data [%s]"%item_class)
            item_list = {}
            for key in range(0, max_value, day):
                item_list[key] = []
                for name in self.df['name'].unique():
                    for c in course_list:
                        bar.numerator += 1
                        if bar.numerator%100 == 0:
                            print("%s" % (bar,), end='\r')
                        try:
                            item_list[key].append(self.data[name][c][item_class][key])
                        except KeyError:
                            print("Key Error: ", name, c, item_class, key)
                try:
                    self.mean_data[item_class][key] = np.mean(item_list[key])
                except ValueError:
                    print("ValueError: ", item_class, key)
                except TypeError:
                    print("TypeError: ", item_class, key)

        make_mean('humidity')
        make_mean('month')
        make_mean('age')
        make_mean_day('hr_days', 30)
        make_mean_day('lastday', 10)
        make_mean('idx')
        make_mean('rcno')
        make_mean('budam')
        make_mean('dbudam')
        make_mean('jockey')
        make_mean('trainer')
        for j in range(1,82):
            make_mean('jc%d'%j)

        print(self.mean_data)



class mean_data2:
    def __init__(self):
        self.mean_pkl = joblib.load('mean_data2.pkl')
        md = self.mean_pkl.mean_data
        fout = open('mean_data2.csv', 'wt')
        fout.write("data..\n")
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
    m = Make_mean()
    m.get_data()
    joblib.dump(m, 'mean_data2.pkl')
    md = mean_data2()


# -*- coding:utf-8 -*-


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


class make_mean:
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
        for i in range(len(self.df)):
            row = self.df.iloc[i,:]

            if row['name'] not in self.data:
                self.data[row['name']] = {'humidity':{}, 'month':{}, 'age':{}, 'idx':{}, 'rcno':{}, 'budam':{}, 'dbudam':{}, 'jockey':{}, 'trainer':{}}
                for j in range(1,82):
                    self.data[row['name']]['jc%d'%j] = {}

            def add_item(data, row, item):
                try:
                    data[row['name']][item][row[item]].append(row['rctime'])
                except KeyError:
                    data[row['name']][item][row[item]] = [row['rctime']]

            add_item(self.data, row, 'humidity')
            add_item(self.data, row, 'month')
            add_item(self.data, row, 'age')
            add_item(self.data, row, 'idx')
            add_item(self.data, row, 'rcno')
            add_item(self.data, row, 'budam')
            add_item(self.data, row, 'dbudam')
            add_item(self.data, row, 'jockey')
            add_item(self.data, row, 'trainer')
            for j in range(1,82):
                add_item(self.data, row, 'jc%d'%j)

        def make_norm(data, name, item):
            average = []
            for hum in self.df[item].unique():
                try:
                    data[name][item][hum] = np.mean(data[name][item][hum])
                    average.append(data[name][item][hum])
                except KeyError:
                    data[name][item][hum] = 1
            average = np.mean(average)
            for hum in self.df[item].unique():
                if data[name][item][hum] != 1:
                    data[name][item][hum] = data[name][item][hum] / average

        for name in self.df['name'].unique():
            make_norm(self.data, name, 'humidity')
            make_norm(self.data, name, 'month')
            make_norm(self.data, name, 'age')
            make_norm(self.data, name, 'idx')
            make_norm(self.data, name, 'rcno')
            make_norm(self.data, name, 'budam')
            make_norm(self.data, name, 'dbudam')
            make_norm(self.data, name, 'jockey')
            make_norm(self.data, name, 'trainer')
            for j in range(1, 82):
                make_norm(self.data, name, 'jc%d'%j)

        self.mean_data = {'humidity':{}, 'month':{}, 'age':{}, 'idx':{}, 'rcno':{}, 'budam':{}, 'dbudam':{}, 'jockey':{}, 'trainer':{}, 'jc1':{}}

        def make_mean(item_class):
            item_list = {}
            for item in self.df[item_class].unique():
                item_list[item] = []
                for name in self.df['name'].unique():
                    item_list[item].append(self.data[name][item_class][item])
                self.mean_data[item_class][item] = np.mean(item_list[item])

        make_mean('humidity')
        make_mean('month')
        make_mean('age')
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
        pass

    def make_humidity(self):
        pass

    def make_data(self):
        pass



if __name__ == '__main__':
    DEBUG = True
    m = make_mean()
    m.get_data()

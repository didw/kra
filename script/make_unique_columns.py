# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.externals import joblib

def make_unique_columns():
    df = pd.read_csv('../data/1_2007_2016_v1.csv')
    name_one_hot_columns = ['course', 'humidity', 'kind', 'idx', 'cntry', 'gender', 'age', 'jockey', 'trainer', 'owner', 'cnt', 'rcno', 'month']
    key_list = {}
    for column in name_one_hot_columns:
        key_list[column] = sorted(df[column].unique())
    joblib.dump(key_list, '../data/column_unique.pkl')

if __name__ == '__main__':
    make_unique_columns()

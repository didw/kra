#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import parse_txt_race as pr
import datetime
import pandas as pd
import os.path
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.externals import joblib
import simulation as sim
from mean_data import mean_data
import numpy as np
import time
from etaprogress.progress import ProgressBar
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

MODEL_NUM = 10

def dense(input, n_in, n_out, p_keep=0.8):
    weights = tf.get_variable('weight', [n_in, n_out], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', [n_out], initializer=tf.constant_initializer(0.0))
    h1 = tf.nn.batch_normalization(input, 0.001, 1.0, 0, 1, 0.0001)
    return tf.nn.dropout(tf.nn.elu(tf.matmul(h1, weights) + biases), p_keep)


def dense_with_onehot(input, n_in, n_out, p_keep=0.8):
    inputs = tf.reshape(tf.one_hot(tf.to_int32(input), depth=n_in, on_value=1.0, off_value=0.0, axis=-1), [-1, n_in])
    weights = tf.get_variable('weight', [n_in, n_out], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', [n_out], initializer=tf.constant_initializer(0.0))
    h1 = tf.nn.batch_normalization(inputs, 0.001, 1.0, 0, 1, 0.0001)
    return tf.nn.dropout(tf.nn.elu(tf.matmul(h1, weights) + biases), p_keep)



idx_input  = [ 1,  1, 1, 19,  1,  1, 1, 1, 1,   1,   1,   1,  2,  1,  1,  1, 136]
is_onehot  = [ 1,  1, 1,  0,  1,  1, 1, 1, 0,   1,   1,   1,  0,  1,  1,  1,   0]
len_onehot = [10, 20, 8,  0, 16, 17, 3, 9, 0, 256, 130, 902,  0, 12, 15, 12,   0]
len_h1s    = [ 2,  2, 2, 10,  2,  2, 2, 2, 1,  50,  50,  50,  2,  3,  3,  3, 100]
name_one_hot_columns = ['course', 'humidity', 'kind', 'idx', 'cntry', 'gender', 'age', 'jockey', 'trainer', 'owner', 'cnt', 'rcno', 'month']

def build_model(input, p_keep):
    inputs = tf.split(input, idx_input, 1)
    h1s = []
    for i in range(len(idx_input)):
        with tf.variable_scope('h1_%d'%i):
            if is_onehot[i] == 1:
                h1s.append(dense_with_onehot(inputs[i], len_onehot[i], len_h1s[i], p_keep))
            else:
                h1s.append(dense(inputs[i], idx_input[i], len_h1s[i], p_keep))
    h1 = tf.concat(h1s, 1)
    """
    with tf.variable_scope('h1'):
        h1 = dense(input, np.sum(idx_input), np.sum(len_h1s))
    """
    with tf.variable_scope('h2'):
        h2 = dense(h1, np.sum(len_h1s), 100, p_keep)

    with tf.variable_scope('h3'):
        h3 = dense(h2, 100, 10, p_keep)

    with tf.variable_scope('h4'):
        weights = tf.get_variable('weight', [10, 1], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0.0))
        h4 = tf.nn.batch_normalization(h3, 0.001, 1.0, 0, 1, 0.0001)
        return tf.matmul(h4, weights) + biases


class TensorflowRegressor():
    def __init__(self, s_date, scaler_y):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        # Create two variables.
        self.scaler_y = scaler_y
        tf.reset_default_graph()
        self.num_epoch = 80
        self.lr = tf.placeholder(dtype=tf.float32)

        self.Input =  tf.placeholder(shape=[None,171],dtype=tf.float32)
        self.p_keep =  tf.placeholder(shape=None,dtype=tf.float32)
        self.output = tf.reshape(build_model(self.Input, self.p_keep), [-1])
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.target = tf.placeholder(shape=[None],dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.target, self.output)
        self.updateModel = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.saver = tf.train.Saver()
        self.model_dir = '../model/tf/l3_e80_d1_0/%s' % s_date

        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()

        self.init_op = tf.global_variables_initializer()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.column_unique = joblib.load('../data/column_unique.pkl')
        self.dir = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def fit(self, X_data, Y_data, X_val=None, Y_val=None):
        # Add an op to initialize the variables.
        batch_size = 512
        lr = 1e-2
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            bar = ProgressBar(len(X_data)/batch_size*self.num_epoch, max_width=80)
            avg_loss, avg_loss_val = 0, 0
            smr_train = tf.summary.FileWriter('TB/%s/train'%self.dir, sess.graph)
            smr_test = tf.summary.FileWriter('TB/%s/test'%self.dir)
            idx = 0
            for i in range(self.num_epoch):
                lr *= 0.99
                #print("\nEpoch %d/%d is started" % (i+1, self.num_epoch), end='\n')
                idx_val = 0
                for j in range(int(len(X_data)/batch_size)-1):
                    X_batch = X_data[batch_size*j:batch_size*(j+1)]
                    Y_batch = Y_data[batch_size*j:batch_size*(j+1)]
                    X_batch, Y_batch = shuffle(X_batch, Y_batch)
                    _ = sess.run(self.updateModel, feed_dict={self.lr:lr, self.Input: X_batch, self.target: Y_batch, self.p_keep: 1.0})

                    bar.numerator += 1
                    if j%50 == 0 and j > 0:
                        idx += 1
                        if idx_val == 0:
                            X_val, Y_val = shuffle(X_val, Y_val)
                        summary, loss = sess.run([self.merged, self.loss], feed_dict={self.lr:lr, self.Input: X_batch, self.target: Y_batch, self.p_keep: 1.0})
                        smr_train.add_summary(summary, idx)
                        X_val_batch = X_val[batch_size*idx_val:batch_size*(idx_val+1)]
                        Y_val_batch = Y_val[batch_size*idx_val:batch_size*(idx_val+1)]
                        idx_val += 1
                        if batch_size*(idx_val+1) > len(X_val):
                            idx_val = 0
                        summary, target, loss_val = sess.run([self.merged, self.output, self.loss], feed_dict={self.lr:lr, self.Input: X_val_batch, self.target: Y_val_batch, self.p_keep: 1.0})
                        smr_test.add_summary(summary, idx)
                        if avg_loss == 0:
                            avg_loss = loss
                            avg_loss_val = loss_val
                        else:
                            avg_loss = 0.9*avg_loss + 0.1*loss
                            avg_loss_val = 0.9*avg_loss_val + 0.1*loss_val
                        t = self.scaler_y.inverse_transform(target[0])
                        y = self.scaler_y.inverse_transform(Y_val_batch[0])
                        print("%s | loss_train: %f, loss_val: %f, course: %d, target: %f, Y: %f" % (bar, avg_loss, avg_loss_val, self.column_unique['course'][int(X_val_batch[0][0])], t, y), end='\r')
                        sys.stdout.flush()

            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            save_path = self.saver.save(sess,'%s/model.ckpt' % self.model_dir)
            print("\nModel saved in file: %s" % save_path)

    def predict(self, X_data):
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return sess.run(self.output, feed_dict={self.Input: X_data, self.p_keep: 1.0})



def normalize_data(org_data):
    data = org_data.dropna()
    data = data.reset_index()

    column_unique = joblib.load('../data/column_unique.pkl')
    for column in name_one_hot_columns:
        for idx, value in enumerate(column_unique[column]):
            try:
                data.loc[data[column]==value, column] = idx
            except TypeError:
                print(column, idx, value)
                raise
    return data

def get_data(begin_date, end_date):
    train_bd = begin_date
    train_ed = end_date
    date = train_bd
    data = pd.DataFrame()
    first = True
    date += datetime.timedelta(days=-1)
    while date < train_ed:
        date += datetime.timedelta(days=1)
        if date.weekday() != 5 and date.weekday() != 6:
            continue
        filename = "../txt/1/rcresult/rcresult_1_%02d%02d%02d.txt" % (date.year, date.month, date.day)
        if not os.path.isfile(filename):
            continue
        if first:
            data = pr.get_data(filename)
            first = False
        else:
            data = data.append(pr.get_data(filename), ignore_index=True)
    print(data)
    data = normalize_data(data)
    print(data['cnt'])
    print(data['rcno'])
    R_data = data[['rank', 'r1', 'r2', 'r3', 'hr_nt', 'hr_dt', 'jk_nt', 'tr_nt', 'cnt', 'rcno']]
    Y_data = data['rctime']
    X_data = data.copy()
    del X_data['name']
    del X_data['jockey']
    del X_data['trainer']
    del X_data['owner']
    del X_data['rctime']
    del X_data['rank']
    del X_data['r3']
    del X_data['r2']
    del X_data['r1']
    del X_data['date']
    print(R_data)
    return X_data, Y_data, R_data, data


def get_data_from_csv(begin_date, end_date, fname_csv, course=0, kind=0, nData=47):
    df = pd.read_csv(fname_csv)
    remove_index = []
    for idx in range(len(df)):
        #print(df['date'][idx])
        date = int(df['date'][idx])
        if date < begin_date or date > end_date or (course > 0 and course != int(df['course'][idx])) or (kind > 0 and kind != int(df['kind'][idx])):
            remove_index.append(idx)
            #print('Delete %dth row (hr: %s, jk: %s, tr: %s, dt: %s)' % (idx, X_data['hr_nt'][idx], X_data['jk_nt'][idx], X_data['tr_nt'][idx], X_data['hr_dt'][idx]))
    data = df.drop(df.index[remove_index])
    data = normalize_data(data)

    R_data = data[['name', 'rank', 'r1', 'r2', 'r3', 'hr_nt', 'hr_dt', 'jk_nt', 'tr_nt', 'cnt', 'rcno', 'price', 'bokyeon1', 'bokyeon2', 'bokyeon3', 'boksik', 'ssang', 'sambok', 'samssang', 'idx']]
    Y_data = data['rctime']
    X_data = data.copy()
    X_data = X_data.drop(['name', 'rctime', 'rank', 'r3', 'r2', 'r1', 'date', 'price', 'bokyeon1', 'bokyeon2', 'bokyeon3', 'boksik', 'ssang', 'sambok', 'ssang', 'samssang', 'index'], axis=1)
    #X_data = X_data.drop(['jockey', 'trainer', 'owner'], axis=1)
    #print(X_data.columns)
    X_data = X_data.drop(['jk%d'%i for i in range(1, 257)] + ['tr%d'%i for i in range(1, 130)], axis=1)
    if nData == 47:
        X_data = X_data.drop(['ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'ts6', 'score1', 'score2', 'score3', 'score4', 'score5', 'score6', 'score7', 'score8', 'score9', 'score10', 'hr_dt', 'hr_d1', 'hr_d2', 'hr_rh', 'hr_rm', 'hr_rl'], axis=1)
        X_data = X_data.drop(['rd1', 'rd2', 'rd3', 'rd4', 'rd5', 'rd6', 'rd7', 'rd8', 'rd9', 'rd10', 'rd11', 'rd12', 'rd13', 'rd14', 'rd15', 'rd16', 'rd17', 'rd18', # 18
                  'jc1', 'jc2', 'jc3', 'jc4', 'jc5', 'jc6', 'jc7', 'jc8', 'jc9', 'jc10', 'jc11', 'jc12', 'jc13', 'jc14', 'jc15', 'jc16', 'jc17', 'jc18', 'jc19', 'jc20', 'jc21', 'jc22', 'jc23', 'jc24', 'jc25', 'jc26', 'jc27', 'jc28', 'jc29', 'jc30',
                  'jc31', 'jc32', 'jc33', 'jc34', 'jc35', 'jc36', 'jc37', 'jc38', 'jc39', 'jc40', 'jc41', 'jc42', 'jc43', 'jc44', 'jc45', 'jc46', 'jc47', 'jc48', 'jc49', 'jc50', 'jc51', 'jc52', 'jc53', 'jc54', 'jc55', 'jc56', 'jc57', 'jc58', 'jc59', 'jc60',
                  'jc61', 'jc62', 'jc63', 'jc64', 'jc65', 'jc66', 'jc67', 'jc68', 'jc69', 'jc70', 'jc71', 'jc72', 'jc73', 'jc74', 'jc75', 'jc76', 'jc77', 'jc78', 'jc79', 'jc80', 'jc81'], axis=1)
    if nData == 105:
        X_data = X_data.drop(['jc1', 'jc2', 'jc3', 'jc4', 'jc5', 'jc6', 'jc7', 'jc8', 'jc9', 'jc10', 'jc11', 'jc12', 'jc13', 'jc14', 'jc15', 'jc16', 'jc17', 'jc18', 'jc19', 'jc20', 'jc21', 'jc22', 'jc23', 'jc24', 'jc25', 'jc26', 'jc27', 'jc28', 'jc29', 'jc30',
                  'jc31', 'jc32', 'jc33', 'jc34', 'jc35', 'jc36', 'jc37', 'jc38', 'jc39', 'jc40', 'jc41', 'jc42', 'jc43', 'jc44', 'jc45', 'jc46', 'jc47', 'jc48', 'jc49', 'jc50', 'jc51', 'jc52', 'jc53', 'jc54', 'jc55', 'jc56', 'jc57', 'jc58', 'jc59', 'jc60',
                  'jc61', 'jc62', 'jc63', 'jc64', 'jc65', 'jc66', 'jc67', 'jc68', 'jc69', 'jc70', 'jc71', 'jc72', 'jc73', 'jc74', 'jc75', 'jc76', 'jc77', 'jc78', 'jc79', 'jc80', 'jc81'], axis=1)
    return X_data, Y_data, R_data, data

def delete_lack_data(X_data, Y_data):
    remove_index = []
    for idx in range(len(X_data)):
        if X_data['hr_nt'][idx] == -1 or X_data['jk_nt'][idx] == -1 or X_data['tr_nt'][idx] == -1:
            remove_index.append(idx)
            #print('Delete %dth row (hr: %s, jk: %s, tr: %s, dt: %s)' % (idx, X_data['hr_nt'][idx], X_data['jk_nt'][idx], X_data['tr_nt'][idx], X_data['hr_dt'][idx]))
    print(len(remove_index))
    return X_data.drop(X_data.index[remove_index]), Y_data.drop(Y_data.index[remove_index])

def training(train_bd, train_ed, course=0, nData=47):
    train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
    train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))

    estimators = [0] * MODEL_NUM
    print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
    X_train, Y_train, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016_v1.csv', 0, nData=nData)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    for i in range(MODEL_NUM):
        print("model[%d] training.." % (i+1))
        dir_name = '../model/tf/l3_e80_d1_0/%s_%s_%d/model.ckpt' % (train_bd, train_ed, i)
        tf.reset_default_graph()
        estimators[i] = TensorflowRegressor("%s_%s_%d"%(train_bd, train_ed, i))
        if os.path.exists("%s/model_ckpt"%dir_name):
            print("loading exists model")
            estimators[i].load("%s/model_ckpt"%dir_name)
        else:
            estimators[i].fit(X_train, Y_train)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        estimators[i].save("%s/model_ckpt"%dir_name)
    print("Finish train model")
    md = joblib.load('../data/1_2007_2016_v1_md.pkl')
    return estimators, md

def print_log(data, pred, fname):
    flog = open(fname, 'w')
    rcno = 1
    flog.write("rcno\tcourse\tidx\tname\tcntry\tgender\tage\tbudam\tjockey\ttrainer\tweight\tdweight\thr_days\thumidity\thr_nt\thr_nt1\thr_nt2\thr_ny\thr_ny1\thr_ny2\t")
    flog.write("jk_nt\tjk_nt1\tjk_nt2\tjk_ny\tjk_ny1\tjk_ny2\ttr_nt\ttr_nt1\ttr_nt2\ttr_ny\ttr_ny1\ttr_ny2\tpredict\n")
    for idx in range(len(data)):
        if rcno != data['rcno'][idx]:
            rcno = data['rcno'][idx]
            flog.write('\n')
        flog.write("%s\t%s\t%s\t%s\t%s\t" % (data['rcno'][idx], data['course'][idx], data['idx'][idx], data['name'][idx], data['cntry'][idx]))
        flog.write("%s\t%s\t%s\t%s\t%s\t" % (data['gender'][idx], data['age'][idx], data['budam'][idx], data['jockey'][idx], data['trainer'][idx]))
        flog.write("%s\t%s\t%s\t%s\t" % (data['weight'][idx], data['dweight'][idx], data['hr_days'][idx], data['humidity'][idx]))
        flog.write("%s\t%s\t%s\t%s\t%s\t%s\t" % (data['hr_nt'][idx], data['hr_nt1'][idx], data['hr_nt2'][idx], data['hr_ny'][idx], data['hr_ny1'][idx], data['hr_ny2'][idx]))
        flog.write("%s\t%s\t%s\t%s\t%s\t%s\t" % (data['jk_nt'][idx], data['jk_nt1'][idx], data['jk_nt2'][idx], data['jk_ny'][idx], data['jk_ny1'][idx], data['jk_ny2'][idx]))
        flog.write("%s\t%s\t%s\t%s\t%s\t%s\t" % (data['tr_nt'][idx], data['tr_nt1'][idx], data['tr_nt2'][idx], data['tr_ny'][idx], data['tr_ny1'][idx], data['tr_ny2'][idx]))
        flog.write("%f\n" % pred['predict'][idx])
    flog.close()


def simulation_weekly_train0(begin_date, end_date, delta_day=0, delta_year=0, courses=[0], kinds=[0], nData=47):
    remove_outlier = False
    today = begin_date
    def m():
        return [{0:0} for _ in range(MODEL_NUM+1)]
    sr1, sr2, sr3, sr4, sr5, sr6, sr7, sr8, sr9, sr10, score_sum = m(),m(),m(),m(),m(),m(),m(),m(),m(),m(),m()

    while today <= end_date:
        while today.weekday() != 3:
            today = today + datetime.timedelta(days=1)
        today = today + datetime.timedelta(days=1)
        train_bd = today + datetime.timedelta(days=-365*delta_year)
        #train_bd = datetime.date(2011, 1, 1)
        train_ed = today + datetime.timedelta(days=-delta_day)
        test_bd = today + datetime.timedelta(days=1)
        test_ed = today + datetime.timedelta(days=2)
        test_bd_s = "%d%02d%02d" % (test_bd.year, test_bd.month, test_bd.day)
        test_ed_s = "%d%02d%02d" % (test_ed.year, test_ed.month, test_ed.day)
        if not os.path.exists('../txt/1/rcresult/rcresult_1_%s.txt' % test_bd_s) and not os.path.exists('../txt/1/rcresult/rcresult_1_%s.txt' % test_ed_s):
            continue
        train_bd_i = int("%d%02d%02d" % (train_bd.year, train_bd.month, train_bd.day))
        train_ed_i = int("%d%02d%02d" % (train_ed.year, train_ed.month, train_ed.day))

        print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
        X_train, Y_train, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016_v1.csv', 0, nData=nData)
        scaler_x1, scaler_x2, scaler_x3, scaler_x4, scaler_y = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_train, Y_train = shuffle(X_train, Y_train)
        
        X_train[:,3:22] = scaler_x1.fit_transform(X_train[:,3:22])
        X_train[:,26:27] = scaler_x2.fit_transform(X_train[:,26:27])
        X_train[:,30:32] = scaler_x3.fit_transform(X_train[:,30:32])
        X_train[:,35:171] = scaler_x4.fit_transform(X_train[:,35:171])
        """
        X_train = scaler_x1.fit_transform(X_train)
        """
        Y_train = scaler_y.fit_transform(Y_train)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, random_state=0)
        for i in range(MODEL_NUM):
            print("model[%d] training.." % (i+1))
            tf.reset_default_graph()
            estimators = TensorflowRegressor("%s_%s_%d"%(train_bd, train_ed, i), scaler_y)
            if not os.path.exists('../model/tf/l3_e80_d1_0/%s_%s_%d/model.ckpt' % (train_bd, train_ed, i)):
                estimators.fit(X_train, Y_train, X_val, Y_val)
        print("Finish train model")

        test_bd_i = int("%d%02d%02d" % (test_bd.year, test_bd.month, test_bd.day))
        test_ed_i = int("%d%02d%02d" % (test_ed.year, test_ed.month, test_ed.day))

        for course in courses:
            for kind in kinds:
                if not os.path.exists('../data/tf/l3_e80_d1_0'):
                    os.makedirs('../data/tf/l3_e80_d1_0')
                fname_result = '../data/tf/l3_e80_d1_0/tf_v1_ss_nd%d_y%d_c%d.txt' % (nData, delta_year, course)
                print("Loading Datadata at %s - %s" % (str(test_bd), str(test_ed)))
                X_test, Y_test, R_test, X_data = get_data_from_csv(test_bd_i, test_ed_i, '../data/1_2007_2016_v1.csv', course, kind, nData=nData)
                X_test = np.array(X_test)
                X_test[:,3:22] = scaler_x1.transform(X_test[:,3:22])
                X_test[:,26:27] = scaler_x2.transform(X_test[:,26:27])
                X_test[:,30:32] = scaler_x3.transform(X_test[:,30:32])
                X_test[:,35:171] = scaler_x4.transform(X_test[:,35:171])
                """
                X_test = scaler_x1.transform(X_test)
                """
                print("%d data is fully loaded" % (len(X_test)))

                print("train data: %s - %s" % (str(train_bd), str(train_ed)))
                print("test data: %s - %s" % (str(test_bd), str(test_ed)))
                print("course: %d[%d]" % (course, kind))
                print("%15s%10s%10s%10s%10s%10s%10s%10s" % ("score", "d", "y", "b", "by", "s", "sb", "ss"))

                res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                if len(X_test) == 0:
                    res1, res2, res3, res4, res5, res6, res7, res8 = 0, 0, 0, 0, 0, 0, 0, 0
                    continue
                else:
                    DEBUG = False
                    if DEBUG:
                        X_test.to_csv('../log/weekly_train0_%s.csv' % today, index=False)
                    X_test = np.array(X_test)
                    Y_test = np.array(Y_test.reshape(-1,1)).reshape(-1)
                    pred = [0] * MODEL_NUM
                    for i in range(MODEL_NUM):
                        estimators = TensorflowRegressor('%s_%s_%d' % (train_bd, train_ed, i), scaler_y)
                        pred[i] = estimators.predict(X_test)
                        pred[i] = scaler_y.inverse_transform(pred[i])
                        score = np.sqrt(np.mean((pred[i] - Y_test)*(pred[i] - Y_test)))

                        res1 = sim.simulation7(pred[i], R_test, [[1],[2],[3]])
                        res2 = sim.simulation7(pred[i], R_test, [[1,2],[1,2,3],[1,2,3]])
                        res3 = sim.simulation7(pred[i], R_test, [[1,2,3],[1,2,3],[1,2,3]])
                        res4 = sim.simulation7(pred[i], R_test, [[1,2,3],[1,2,3,4,5],[1,2,3,4,5,6]])
                        res5 = sim.simulation7(pred[i], R_test, [[3,4,5],[4,5,6],[4,5,6]])
                        res6 = sim.simulation7(pred[i], R_test, [[4,5,6],[4,5,6],[4,5,6,7]])
                        res7 = sim.simulation7(pred[i], R_test, [[4,5,6,7],[4,5,6,7],[4,5,6,7]])
                        
                        print("pred[%d] test: " % (i+1), pred[i][0:4])
                        print("Y_test[%d] test: " % (i+1), Y_test[0:4])
                        print("result[%d]: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f\n" % (
                                i+1, score, res1, res2, res3, res4, res5, res6, res7))
                        try:
                            sr1[i][course] += res1
                            sr2[i][course] += res2
                            sr3[i][course] += res3
                            sr4[i][course] += res4
                            sr5[i][course] += res5
                            sr6[i][course] += res6
                            sr7[i][course] += res7
                            score_sum[i][course] += score
                        except KeyError:
                            sr1[i][course] = res1
                            sr2[i][course] = res2
                            sr3[i][course] = res3
                            sr4[i][course] = res4
                            sr5[i][course] = res5
                            sr6[i][course] = res6
                            sr7[i][course] = res7
                            score_sum[i][course] = score
                    pred = np.mean(pred, axis=0)
                    print("pred test: ", pred[0:4])
                    print("Y_test test: ", Y_test[0:4])
                    score = np.sqrt(np.mean((pred - Y_test)*(pred - Y_test)))

                    res1 = sim.simulation7(pred, R_test, [[1],[2],[3]])
                    res2 = sim.simulation7(pred, R_test, [[1,2],[1,2,3],[1,2,3]])
                    res3 = sim.simulation7(pred, R_test, [[1,2,3],[1,2,3],[1,2,3]])
                    res4 = sim.simulation7(pred, R_test, [[1,2,3],[1,2,3,4,5],[1,2,3,4,5,6]])
                    res5 = sim.simulation7(pred, R_test, [[3,4,5],[4,5,6],[4,5,6]])
                    res6 = sim.simulation7(pred, R_test, [[4,5,6],[4,5,6],[4,5,6,7]])
                    res7 = sim.simulation7(pred, R_test, [[4,5,6,7],[4,5,6,7],[4,5,6,7]])
                    
                    """
                    res1 = sim.simulation1(pred, R_test, 1)
                    res2 = sim.simulation2(pred, R_test, 1)
                    res3 = sim.simulation3(pred, R_test, [[1,2]])
                    res4 = sim.simulation4(pred, R_test, [1,2])
                    res5 = sim.simulation5(pred, R_test, [[1,2]])
                    res6 = sim.simulation6(pred, R_test, [[1,2,3]])
                    res7 = sim.simulation7(pred, R_test, [[1,2,3],[1,2,3],[2,3,4]])

                    res1 = sim.simulation5(pred, R_test, [[1,2]])
                    res2 = sim.simulation5(pred, R_test, [[1,2],[1,3]])
                    res3 = sim.simulation5(pred, R_test, [[1,2],[1,3],[2,3]])
                    res4 = sim.simulation5(pred, R_test, [[1,2],[1,3],[2,3],[1,4]])
                    res5 = sim.simulation5(pred, R_test, [[1,2],[1,3],[2,3],[1,4],[2,4],[3,4]])
                    res6 = sim.simulation5(pred, R_test, [[1,2],[1,3],[2,3],[1,4],[2,4],[3,4],[1,5]])
                    res7 = sim.simulation5(pred, R_test, [[1,2],[1,3],[2,3],[1,4],[2,4],[3,4],[1,5],[2,5],[3,5],[4,5]])
                    
                    res1 = sim.simulation6(pred, R_test, [[1,2,3]])
                    res2 = sim.simulation6(pred, R_test, [[1,2,3], [1,2,4], [1,3,4], [2,3,4]])
                    res3 = sim.simulation6(pred, R_test, [[1,2,3], [1,2,4], [1,2,5], [1,3,4], [1,3,5], [1,4,5], [2,3,4], [2,3,5], [2,4,5], [3,4,5]])
                    res4 = sim.simulation6(pred, R_test, [[1,2,3], [1,2,4], [1,2,5], [1,3,4], [1,3,5], [1,4,5], [2,3,4], [2,3,5], [2,4,5], [3,4,5],
                                                        [1,2,6], [1,3,6], [1,4,6], [1,5,6], [2,3,6], [2,4,6], [2,5,6], [3,4,6], [3,5,6], [4,5,6]])
                    res5 = sim.simulation6(pred, R_test, [[1,2,3], [1,2,4], [1,2,5], [1,3,4], [1,3,5], [1,4,5], [2,3,4], [2,3,5], [2,4,5], [3,4,5],
                                                        [1,2,6], [1,3,6], [1,4,6], [1,5,6], [2,3,6], [2,4,6], [2,5,6], [3,4,6], [3,5,6], [4,5,6],
                                                        [1,2,7], [1,3,7], [1,4,7], [1,5,7], [2,3,7], [2,4,7], [2,5,7], [3,4,7], [3,5,7], [4,5,7],
                                                        [1,6,7], [2,6,7], [3,6,7], [4,6,7], [5,6,7]
                                                        ])
                    res6 = sim.simulation6(pred, R_test, [[2,3,4], [2,3,5], [2,4,5], [3,4,5], [2,3,6], [2,4,6], [2,5,6], [3,4,6], [3,5,6], [4,5,6]])
                    res7 = sim.simulation6(pred, R_test, [[3,4,5], [3,4,6], [3,4,7], [3,5,6], [3,5,7], [3,6,7], [4,5,6], [4,5,7], [4,6,7], [5,6,7]])
                    
                    res1 = sim.simulation7(pred, R_test, [[1],[2],[3]])
                    res2 = sim.simulation7(pred, R_test, [[1,2],[1,2,3],[1,2,3]])
                    res3 = sim.simulation7(pred, R_test, [[1,2,3],[1,2,3,4,5],[1,2,3,4,5,6]])
                    res4 = sim.simulation7(pred, R_test, [[1,2,3,4],[1,2,3,4,5,6],[3,4,5,6]])
                    res5 = sim.simulation7(pred, R_test, [[4,5,6],[4,5,6],[4,5,6]])
                    res6 = sim.simulation7(pred, R_test, [[4,5,6,7,8],[4,5,6,7,8],[4,5,6,7,8]])
                    res7 = sim.simulation7(pred, R_test, [[5,6,7,8,9,10],[5,6,7,8,9,10],[5,6,7,8,9,10]])
                    
                    res1 = sim.simulation2(pred, R_test, 1)
                    res2 = sim.simulation2(pred, R_test, 2)
                    res3 = sim.simulation2(pred, R_test, 3)
                    res4 = sim.simulation2(pred, R_test, 4)
                    res5 = sim.simulation2(pred, R_test, 5)
                    res6 = sim.simulation2(pred, R_test, 6)
                    res7 = sim.simulation2(pred, R_test, 7)
                    """
                    
                    try:
                        sr1[MODEL_NUM][course] += res1
                        sr2[MODEL_NUM][course] += res2
                        sr3[MODEL_NUM][course] += res3
                        sr4[MODEL_NUM][course] += res4
                        sr5[MODEL_NUM][course] += res5
                        sr6[MODEL_NUM][course] += res6
                        sr7[MODEL_NUM][course] += res7
                        score_sum[MODEL_NUM][course] += score
                    except KeyError:
                        sr1[MODEL_NUM][course] = res1
                        sr2[MODEL_NUM][course] = res2
                        sr3[MODEL_NUM][course] = res3
                        sr4[MODEL_NUM][course] = res4
                        sr5[MODEL_NUM][course] = res5
                        sr6[MODEL_NUM][course] = res6
                        sr7[MODEL_NUM][course] = res7
                        score_sum[MODEL_NUM][course] = score

                print("result: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f\n" % (
                        score, res1, res2, res3, res4, res5, res6, res7))
                for m in range(MODEL_NUM+1):
                    print("result: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f\n" % (
                            score_sum[m][course], sr1[m][course], sr2[m][course], sr3[m][course], sr4[m][course], sr5[m][course], sr6[m][course], sr7[m][course]))
                f_result = open(fname_result, 'a')
                f_result.write("train data: %s - %s\n" % (str(train_bd), str(train_ed)))
                f_result.write("test data: %s - %s\n" % (str(test_bd), str(test_ed)))
                f_result.write("%15s%10s%10s%10s%10s%10s%10s%10s\n" % ("score", "d", "y", "b", "by", "s", "sb", "ss"))
                f_result.write("result: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f\n" % (
                                score, res1, res2, res3, res4, res5, res6, res7))
                f_result.close()
    for m in range(MODEL_NUM+1):
        for course in courses:
            for kind in kinds:
                fname_result = '../data/tf/l3_e80_d1_0/weekly_tf_nsb_v1_ss_train0_m1_nd%d_y%d_c%d_k%d.txt' % (nData, delta_year, course, kind)
                f_result = open(fname_result, 'a')
                f_result.write("%15s%10s%10s%10s%10s%10s%10s%10s\n" % ("score", "d", "y", "b", "by", "s", "sb", "ss"))
                f_result.write("result: %4.5f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f,%9.0f\n" % (
                                score_sum[m][course], sr1[m][course], sr2[m][course], sr3[m][course], sr4[m][course], sr5[m][course], sr6[m][course], sr7[m][course]))
                f_result.close()


if __name__ == '__main__':
    delta_year = 4
    dbname = '../data/train_201101_20160909.pkl'
    train_bd = datetime.date(2011, 11, 1)
    train_ed = datetime.date(2016, 10, 31)
    test_bd = datetime.date(2016, 6, 5)
    test_ed = datetime.date(2017, 3, 13)

    for delta_year in [6]:
        for nData in [186]:
            simulation_weekly_train0(test_bd, test_ed, 0, delta_year, courses=[0], nData=nData)
            #for c in [1000, 1200, 1300, 1400, 1700]:
            #    for k in [0]:
            #        outfile = '../data/weekly_keras_m1_nd%d_y%d_c%d_0_k%d.txt' % (nData, delta_year, c, k)
            #        simulation_weekly(test_bd, test_ed, outfile, 0, delta_year, c, k, nData=nData)

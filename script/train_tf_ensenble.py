#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import parse_txt_race as pr
import datetime
import pandas as pd
import os.path
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import simulation as sim
from mean_data import mean_data
import numpy as np
import time
from etaprogress.progress import ProgressBar
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

MODEL_NUM = 5

class TensorflowRegressor():
    def __init__(self, s_date):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        # Create two variables.
        tf.reset_default_graph()
        self.num_epoch = 200
        self.lr = tf.placeholder(dtype=tf.float32)
        #self.W1 = tf.Variable(tf.random_normal([168, 128], stddev=0.35), name="W1")
        self.W1 = tf.get_variable('W1', shape=(168, 128), initializer=tf.contrib.layers.xavier_initializer()) 
        self.b1 = tf.Variable(tf.zeros([128]), name="b1")
        #self.W2 = tf.Variable(tf.random_normal([128, 1], stddev=0.35), name="W2")
        self.W2 = tf.get_variable('W2', shape=(128, 1), initializer=tf.contrib.layers.xavier_initializer()) 
        self.b2 = tf.Variable(tf.zeros([1]), name="b2")

        self.scalarInput =  tf.placeholder(shape=[None,168],dtype=tf.float32)
        self.out1 = tf.matmul(self.scalarInput, self.W1) + self.b1
        self.stream1 = tf.nn.relu(self.out1)
        #self.output = tf.layers.dense(self.stream1, 1)
        self.output = tf.matmul(self.stream1, self.W2) + self.b2
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.target = tf.placeholder(shape=[None],dtype=tf.float32)
        self.error = tf.square(self.target - self.output)
        self.loss = tf.reduce_mean(self.error)
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay = 0.9)
        self.updateModel = self.trainer.minimize(self.loss)
        self.saver = tf.train.Saver([self.W1, self.b1, self.W2, self.b2])
        self.model_dir = '../model/tf/regression/%s' % s_date

        self.init_op = tf.global_variables_initializer()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

    def fit(self, X_data, Y_data):
        # Add an op to initialize the variables.
        batch_size = 64
        lr = 0.001
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            bar = ProgressBar(len(X_data)/batch_size*self.num_epoch, max_width=80)
            for i in range(self.num_epoch):
                lr *= 0.9
                #print("\nEpoch %d/%d is started" % (i+1, self.num_epoch), end='\n')
                for j in range(int(len(X_data)/batch_size)-1):
                    X_batch = X_data[batch_size*j:batch_size*(j+1)]
                    Y_batch = Y_data[batch_size*j:batch_size*(j+1)]
                    _ = sess.run(self.updateModel, feed_dict={self.lr:lr, self.scalarInput: X_batch, self.target: Y_batch})

                    bar.numerator += 1
                    if j%10 == 0:
                        loss = sess.run(self.loss, feed_dict={self.lr:lr, self.scalarInput: X_batch, self.target: Y_batch})
                        print("%s | loss: %f" % (bar, loss), end='\r')
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
            return sess.run(self.output, feed_dict={self.scalarInput: X_data})


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(128, input_dim=168, init='he_normal', activation='relu'))
    #model.add(Dense(128, input_dim=87, init='he_normal', activation='relu'))
    #model.add(Dropout(0.1))
    #model.add(Dense(128, init='he_normal'))
    model.add(Dense(1, init='he_normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def normalize_data(org_data):
    data = org_data.dropna()
    data = data.reset_index()
    data.loc[data['gender'] == '암', 'gender'] = 0
    data.loc[data['gender'] == '수', 'gender'] = 1
    data.loc[data['gender'] == '거', 'gender'] = 2
    data.loc[data['cntry'] == '한', 'cntry'] = 0
    data.loc[data['cntry'] == '한(포)', 'cntry'] = 1
    data.loc[data['cntry'] == '일', 'cntry'] = 2
    data.loc[data['cntry'] == '중', 'cntry'] = 3
    data.loc[data['cntry'] == '미', 'cntry'] = 4
    data.loc[data['cntry'] == '캐', 'cntry'] = 5
    data.loc[data['cntry'] == '뉴', 'cntry'] = 6
    data.loc[data['cntry'] == '호', 'cntry'] = 7
    data.loc[data['cntry'] == '브', 'cntry'] = 8
    data.loc[data['cntry'] == '헨', 'cntry'] = 9
    data.loc[data['cntry'] == '남', 'cntry'] = 10
    data.loc[data['cntry'] == '아일', 'cntry'] = 11
    data.loc[data['cntry'] == '모', 'cntry'] = 12
    data.loc[data['cntry'] == '영', 'cntry'] = 13
    data.loc[data['cntry'] == '인', 'cntry'] = 14
    data.loc[data['cntry'] == '아', 'cntry'] = 15
    data.loc[data['cntry'] == '프', 'cntry'] = 16
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
    X_data = X_data.drop(['name', 'jockey', 'trainer', 'owner', 'rctime', 'rank', 'r3', 'r2', 'r1', 'date', 'price', 'bokyeon1', 'bokyeon2', 'bokyeon3', 'boksik', 'ssang', 'sambok', 'ssang', 'samssang', 'index'], axis=1)
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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    #os.system('rm -r \"../model_tf/%d_%d/\"' % (train_bd_i, train_ed_i))
    os.system('mkdir \"../model_tf/%d_%d/\"' % (train_bd_i, train_ed_i))
    model_name = "../model_tf/%d_%d/model_v1.h5" % (train_bd_i, train_ed_i)
    md_name = "../model_tf/%d_%d/md_%d.pkl" % (train_bd_i, train_ed_i, course)
    estimators = [0] * MODEL_NUM
    print("Loading Datadata at %s - %s" % (str(train_bd), str(train_ed)))
    X_train, Y_train, _, _ = get_data_from_csv(train_bd_i, train_ed_i, '../data/1_2007_2016_v1.csv', 0, nData=nData)
    print("%d data is fully loaded" % len(X_train))
    #X_scaler = StandardScaler()
    #X_train = X_scaler.fit_transform(X_train)
    print("Start train model")
    # fix random seed for reproducibility
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    for i in range(MODEL_NUM):
        if os.path.exists(model_name.replace('h5', '%d.h5'%i)):
            print("model[%d] exist. try to loading.. %s - %s" % (i, str(train_bd), str(train_ed)))
            estimators[i] = model_from_json(open(model_name.replace('h5', 'json')).read())
            estimators[i].load_weights(model_name.replace('h5', '%d.h5'%i))
        else:
            print("model[%d] training.." % (i+1))
            estimators[i] = KerasRegressor(build_fn=baseline_model, nb_epoch=200, batch_size=32, verbose=0)
            estimators[i].fit(X_train, Y_train)
            # saving model
            json_model = estimators[i].model.to_json()
            open(model_name.replace('h5', 'json'), 'w').write(json_model)
            estimators[i].model.save_weights(model_name.replace('h5', '%d.h5'%i), overwrite=True)
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
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        for i in range(MODEL_NUM):
            print("model[%d] training.." % (i+1))
            estimators = TensorflowRegressor("%s_%s/%d" % (train_bd, train_ed, i))
            estimators.fit(X_train, Y_train)
        print("Finish train model")

        test_bd_i = int("%d%02d%02d" % (test_bd.year, test_bd.month, test_bd.day))
        test_ed_i = int("%d%02d%02d" % (test_ed.year, test_ed.month, test_ed.day))

        for course in courses:
            for kind in kinds:
                fname_result = '../data/weekly_tf_nsb_v1_ss_train0_m1_nd%d_y%d_c%d_k%d.txt' % (nData, delta_year, course, kind)
                print("Loading Datadata at %s - %s" % (str(test_bd), str(test_ed)))
                X_test, Y_test, R_test, X_data = get_data_from_csv(test_bd_i, test_ed_i, '../data/1_2007_2016_v1.csv', course, kind, nData=nData)
                #X_test = X_scaler.transform(X_test)
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
                    Y_test = np.array(Y_test)
                    pred = [0] * MODEL_NUM
                    for i in range(MODEL_NUM):
                        estimators = TensorflowRegressor("%s_%s/%d" % (train_bd, train_ed, i))
                        pred[i] = estimators.predict(X_test).flatten()
                        score = np.sqrt(np.mean((pred[i] - Y_test)*(pred[i] - Y_test)))

                        res1 = sim.simulation7(pred[i], R_test, [[1],[2],[3]])
                        res2 = sim.simulation7(pred[i], R_test, [[1,2],[1,2,3],[1,2,3]])
                        res3 = sim.simulation7(pred[i], R_test, [[1,2,3],[1,2,3,4,5],[1,2,3,4,5,6]])
                        res4 = sim.simulation7(pred[i], R_test, [[1,2,3,4],[1,2,3,4,5,6],[3,4,5,6]])
                        res5 = sim.simulation7(pred[i], R_test, [[4,5,6],[4,5,6],[4,5,6]])
                        res6 = sim.simulation7(pred[i], R_test, [[4,5,6,7,8],[4,5,6,7,8],[4,5,6,7,8]])
                        res7 = sim.simulation7(pred[i], R_test, [[5,6,7,8,9,10],[5,6,7,8,9,10],[5,6,7,8,9,10]])
                        
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
                    res3 = sim.simulation7(pred, R_test, [[1,2,3],[1,2,3,4,5],[1,2,3,4,5,6]])
                    res4 = sim.simulation7(pred, R_test, [[1,2,3,4],[1,2,3,4,5,6],[3,4,5,6]])
                    res5 = sim.simulation7(pred, R_test, [[4,5,6],[4,5,6],[4,5,6]])
                    res6 = sim.simulation7(pred, R_test, [[4,5,6,7,8],[4,5,6,7,8],[4,5,6,7,8]])
                    res7 = sim.simulation7(pred, R_test, [[5,6,7,8,9,10],[5,6,7,8,9,10],[5,6,7,8,9,10]])
                    
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
                fname_result = '../data/weekly_tf_nsb_v1_ss_train0_m1_nd%d_y%d_c%d_k%d.txt' % (nData, delta_year, course, kind)
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
    test_bd = datetime.date(2016, 6, 10)
    test_ed = datetime.date(2017, 3, 13)

    for delta_year in [6]:
        for nData in [186]:
            simulation_weekly_train0(test_bd, test_ed, 0, delta_year, courses=[0], nData=nData)
            #for c in [1000, 1200, 1300, 1400, 1700]:
            #    for k in [0]:
            #        outfile = '../data/weekly_keras_m1_nd%d_y%d_c%d_0_k%d.txt' % (nData, delta_year, c, k)
            #        simulation_weekly(test_bd, test_ed, outfile, 0, delta_year, c, k, nData=nData)

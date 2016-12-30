# -*- coding:utf-8 -*-

from urllib2 import urlopen
import os
from bs4 import BeautifulSoup
import datetime
import re
import numpy as np
import pandas as pd
from sklearn.externals import joblib


DEBUG = False

class mean_data:
    def __init__(self):
        self.lr = 0.001

        self.race_score = \
            {900: [647, 647, 641, 643, 643, 641, 644, 643, 640, 639, 638, 638, 641, 640, 635, 637, 637, 634, 635, 632, 641],
            1000: [647, 647, 641, 643, 643, 641, 644, 643, 640, 639, 638, 638, 641, 640, 635, 637, 637, 634, 635, 632, 641],
            1200: [782, 782, 776, 778, 780, 774, 777, 776, 776, 776, 776, 775, 777, 772, 774, 767, 769, 764, 764, 760, 775],
            1300: [851, 851, 841, 839, 844, 843, 847, 841, 841, 845, 842, 838, 841, 840, 833, 833, 833, 820, 834, 831, 841],
            1400: [913, 913, 906, 909, 910, 907, 908, 908, 908, 907, 907, 905, 906, 903, 896, 890, 896, 896, 883, 890, 906],
            1600: [1047, 1047, 1039, 1044, 1043, 1047, 1041, 1039, 1038, 1040, 1044, 1034, 1042, 1037, 1035, 1026, 1019, 1027, 1029, 1021, 1039],
            0: [815, 815, 807, 809, 811, 809, 810, 808, 807, 808, 808, 805, 808, 805, 801, 798, 798, 796, 796, 794, 807]
            }
        self.course_record = [641, 641, 775, 841, 906, 1039, 807]

        self.dist_rec = {1000: [1.653456998, 2.692917369, 7.338034647, 434.9249578, 439.0969646, 443.8544381],
                         1200: [2.84861923, 6.501179198, 14.04423503, 603.3959383, 610.5974602, 619.0856833],
                         1300: [1.458847737, 4.991679527, 11.47897377, 545.7366255, 550.2887088, 555.3464506],
                         1400: [2.843335238, 9.097669397, 17.97415381, 705.305836, 713.1053534, 721.7971042],
                         1500: [0.862083333, 4.805541667, 10.97866667, 525.6341667, 528.63625, 531.9379167],
                         1600: [2.863417982, 9.358809687, 18.47238945, 784.7052652, 793.2986567, 802.9949995],
                         1800: [3.279455024, 9.700897972, 19.48810961, 906.1184394, 916.5466326, 927.6091191],
                         1900: [1.355569948, 8.175647668, 16.04255181, 757.3646373, 763.2914508, 770.5220207],
                         2000: [3.969792606, 10.70243463, 19.88155996, 983.4481515, 995.5369702, 1007.820559],
                         2200: [1.0, 1.697894737, 6.26, 983.4481515, 995.5369702, 1007.820559]
                         }
        self.hr_days = {1000: 1094.0,
                        1200: 1252.0,
                        1300: 1320.0,
                        1400: 1437.0,
                        1500: 1442.0,
                        1600: 1552.0,
                        1800: 1679.0,
                        1900: 1796.0,
                        2000: 1827.0,
                        2200: 2054.0}
        self.hr_history_total = {1000: [3.535183198, 0.131457918, 0.206423425, 2.639506362, 4.478154224], #nt, nt1, nt2, t1, t2
                                 1200: [8.207720218, 0.587381576, 0.699103003, 8.127645636, 8.533612175],
                                 1300: [9.956532922, 0.657150206, 0.822788066, 9.10622428, 9.086162551],
                                 1400: [13.66158633, 1.43557503, 1.43297136, 12.81920366, 11.11252937],
                                 1500: [14.24541667, 1.227916667, 1.515, 12.43083333, 12.45541667],
                                 1600: [17.45416217, 2.287675262, 2.104422002, 15.94930876, 12.93587607],
                                 1800: [21.08933271, 3.173401455, 2.80647159, 18.02693915, 14.01734015],
                                 1900: [23.94494819, 4.047927461, 3.42746114, 19.74676166, 14.95401554],
                                 2000: [25.14156898, 4.891343553, 3.929666366, 21.87871957, 15.28494139],
                                 2200: [29.98947368, 5.810526316, 4.852631579, 22.89473684, 15.92631579]
                                 }

        self.hr_history_year = {1000: [3.080867699, 0.102023609, 0.175456078, 2.583090602, 4.453012418], #ny, ny1, ny2, y1, y2
                                1200: [6.248337029, 0.417254586, 0.516125781, 7.883188873, 8.326748639],
                                1300: [7.448945473, 0.499356996, 0.608667695, 8.958333333, 8.881687243],
                                1400: [8.885629009, 0.844795834, 0.888423192, 11.83076141, 10.58379374],
                                1500: [9.430833333, 0.835833333, 1.000833333, 12.78291667, 11.96666667],
                                1600: [9.760172566, 1.179527405, 1.126188842, 13.98931268, 12.00784391],
                                1800: [9.966248645, 1.36832327, 1.210249265, 15.03437065, 12.34107447],
                                1900: [9.758419689, 1.381476684, 1.232512953, 15.46761658, 12.52849741],
                                2000: [9.300270514, 1.545085663, 1.27276826, 16.78043282, 13.13300271],
                                2200: [9.526315789, 1.642105263, 1.221052632, 17.67368421, 12.78947368]
                                }

        self.jk_history_total = {1000: [1227.194542, 122.0043692, 120.4591446, 8.511267822, 8.656369769],
                                 1200: [1281.17174, 129.2372506, 127.0502923, 8.527514614, 8.710894981],
                                 1300: [1494.100952, 150.1947016, 148.1756687, 8.471707819, 8.81725823],
                                 1400: [1221.602591, 123.7267416, 121.3939798, 8.556169429, 8.692131835],
                                 1500: [1453.519583, 147.2995833, 144.8220833, 8.558333333, 8.735833333],
                                 1600: [1181.48593, 120.4140602, 117.6834003, 8.653985685, 8.668496911],
                                 1800: [1241.943799, 129.4590494, 125.6807555, 8.904474377, 8.913454095],
                                 1900: [1611.427461, 165.5006477, 161.6353627, 9.057642487, 9.008419689],
                                 2000: [1205.345356, 126.9887286, 122.3119928, 9.000901713, 8.744364292],
                                 2200: [1212.831579, 127.0631579, 123.2947368, 9.778947368, 9.442105263]
                                 }
        self.jk_history_year = {1000: [243.9190556, 25.25095815, 24.28062241, 8.400352598, 8.464893454],
                                1200: [246.5613284, 25.92068131, 24.83183834, 8.47041927, 8.544849829],
                                1300: [253.8459362, 26.71283436, 25.51131687, 8.456018519, 8.607124486],
                                1400: [245.2625262, 25.74426875, 24.77627485, 8.576871785, 8.633453991],
                                1500: [259.75375, 27.93958333, 26.27541667, 8.77375, 8.655833333],
                                1600: [247.2401216, 26.25169134, 24.99362683, 8.756054515, 8.619178351],
                                1800: [259.3517572, 28.33813284, 26.74438768, 9.139340455, 8.920266295],
                                1900: [276.3406736, 29.97797927, 28.25971503, 9.142487047, 8.878238342],
                                2000: [258.1596032, 28.78358882, 26.79891794, 9.288548242, 8.80883679],
                                2200: [278.2421053, 29.66315789, 28.86315789, 9.494736842, 9.094736842]
                                }

        self.tr_history_total = {1000: [1599.698682, 149.1855741, 149.7321784, 8.36762226, 8.546527671],
                                 1200: [1739.2436, 164.7346301, 163.2653195, 8.488560774, 8.572566015],
                                 1300: [2107.46965, 199.0934928, 196.591178, 8.472736626, 8.53279321],
                                 1400: [1692.344383, 166.4648504, 161.7536039, 8.804407189, 8.754937448],
                                 1500: [2089.352083, 202.2370833, 197.73375, 8.621666667, 8.642083333],
                                 1600: [1601.680851, 162.2559074, 155.1833513, 9.013040494, 8.813216982],
                                 1800: [1651.000619, 174.63756, 163.5332095, 9.345564329, 8.96082985],
                                 1900: [2093.076425, 226.6742228, 210.0556995, 9.722150259, 9.215673575],
                                 2000: [1488.363841, 164.2682597, 151.2655546, 9.447249775, 8.940486925],
                                 2200: [1670.894737, 192.2, 172.5894737, 10.10526316, 9.578947368]
                                 }
        self.tr_history_year = {1000: [275.0674536, 25.12647555, 25.39422045, 8.128085237, 8.462517247],
                                1200: [276.0366358, 25.8081536, 25.63802661, 8.343630316, 8.51199355],
                                1300: [275.4180813, 25.94071502, 25.7494856, 8.387602881, 8.531378601],
                                1400: [281.863212, 27.62240427, 26.8030736, 8.782053725, 8.728964247],
                                1500: [281.7408333, 27.16958333, 26.83375, 8.709166667, 8.735833333],
                                1600: [284.2735562, 29.09049907, 27.65467203, 9.110893225, 8.864006275],
                                1800: [289.3170769, 31.33937142, 28.85756309, 9.648552407, 9.106982505],
                                1900: [296.4481865, 33.32059585, 30.51748705, 10.18393782, 9.541450777],
                                2000: [287.7213706, 32.33092876, 29.59963931, 9.820108206, 9.155545537],
                                2200: [305.5578947, 34.87368421, 31.73684211, 10.4, 9.757894737]
                                }
    def get_dist_rec(self):
        return self.dist_rec

    def update_hr(self, course, data):
        self.hr_days[course] += self.lr * (data['hr_days'] - self.hr_days[course])
        self.hr_history_total[course][0] += self.lr * (data['hr_nt'] - self.hr_history_total[course][0])
        self.hr_history_total[course][1] += self.lr * (data['hr_nt1'] - self.hr_history_total[course][1])
        self.hr_history_total[course][2] += self.lr * (data['hr_nt2'] - self.hr_history_total[course][2])
        self.hr_history_total[course][3] += self.lr * (data['hr_t1'] - self.hr_history_total[course][3])
        self.hr_history_total[course][4] += self.lr * (data['hr_t2'] - self.hr_history_total[course][4])
        self.hr_history_year[course][0] += self.lr * (data['hr_ny'] - self.hr_history_year[course][0])
        self.hr_history_year[course][1] += self.lr * (data['hr_ny1'] - self.hr_history_year[course][1])
        self.hr_history_year[course][2] += self.lr * (data['hr_ny2'] - self.hr_history_year[course][2])
        self.hr_history_year[course][3] += self.lr * (data['hr_y1'] - self.hr_history_year[course][3])
        self.hr_history_year[course][4] += self.lr * (data['hr_y2'] - self.hr_history_year[course][4])

    def update_jk(self, course, data):
        self.jk_history_total[course][0] += self.lr * (data['jk_nt']  - self.jk_history_total[course][0])
        self.jk_history_total[course][1] += self.lr * (data['jk_nt1'] - self.jk_history_total[course][1])
        self.jk_history_total[course][2] += self.lr * (data['jk_nt2'] - self.jk_history_total[course][2])
        self.jk_history_total[course][3] += self.lr * (data['jk_t1']  - self.jk_history_total[course][3])
        self.jk_history_total[course][4] += self.lr * (data['jk_t2']  - self.jk_history_total[course][4])
        self.jk_history_year[course][0] += self.lr * (data['jk_ny']  - self.jk_history_year[course][0])
        self.jk_history_year[course][1] += self.lr * (data['jk_ny1'] - self.jk_history_year[course][1])
        self.jk_history_year[course][2] += self.lr * (data['jk_ny2'] - self.jk_history_year[course][2])
        self.jk_history_year[course][3] += self.lr * (data['jk_y1']  - self.jk_history_year[course][3])
        self.jk_history_year[course][4] += self.lr * (data['jk_y2']  - self.jk_history_year[course][4])

    def update_tr(self, course, data):
        self.tr_history_total[course][0] += self.lr * (data['tr_nt']  - self.tr_history_total[course][0])
        self.tr_history_total[course][1] += self.lr * (data['tr_nt1'] - self.tr_history_total[course][1])
        self.tr_history_total[course][2] += self.lr * (data['tr_nt2'] - self.tr_history_total[course][2])
        self.tr_history_total[course][3] += self.lr * (data['tr_t1']  - self.tr_history_total[course][3])
        self.tr_history_total[course][4] += self.lr * (data['tr_t2']  - self.tr_history_total[course][4])
        self.tr_history_year[course][0] += self.lr * (data['tr_ny']  - self.tr_history_year[course][0])
        self.tr_history_year[course][1] += self.lr * (data['tr_ny1'] - self.tr_history_year[course][1])
        self.tr_history_year[course][2] += self.lr * (data['tr_ny2'] - self.tr_history_year[course][2])
        self.tr_history_year[course][3] += self.lr * (data['tr_y1']  - self.tr_history_year[course][3])
        self.tr_history_year[course][4] += self.lr * (data['tr_y2']  - self.tr_history_year[course][4])

    def update_dist_rec(self, course, data):
        self.dist_rec[course][0] += self.lr * (data['hr_dt'] - self.dist_rec[course][0])
        self.dist_rec[course][1] += self.lr * (data['hr_d1'] - self.dist_rec[course][1])
        self.dist_rec[course][2] += self.lr * (data['hr_d2'] - self.dist_rec[course][2])
        self.dist_rec[course][3] += self.lr * (data['hr_rh'] - self.dist_rec[course][3])
        self.dist_rec[course][4] += self.lr * (data['hr_rm'] - self.dist_rec[course][4])
        self.dist_rec[course][5] += self.lr * (data['hr_rl'] - self.dist_rec[course][5])

    def update_race_score_qual(self, humidity, record):
        if humidity > 20:
            humidity = 20
        humidity -= 1
        try:
            self.race_score[900][humidity] += self.lr*10 * (record - self.race_score[900][humidity])
            self.race_score[900][20] = np.mean(self.race_score[900])
        except KeyError:
            return 0

    def update_race_score(self, course, humidity, data):
        humidity = min(humidity, 20) - 1
        try:
            self.race_score[course][humidity] += self.lr * (data['rctime'] - self.race_score[course][humidity])
            self.race_score[course][20] = np.mean(self.race_score[course])
        except KeyError:
            return 0

    def update_data(self, df):
        if DEBUG == True:
            fout = open('../log/md.log', 'w')
        for idx, row in df.iterrows():
            self.update_hr(int(row['course']), row)
            self.update_jk(int(row['course']), row)
            self.update_tr(int(row['course']), row)
            self.update_dist_rec(int(row['course']), row)
            self.update_race_score(int(row['course']), int(row['humidity']), row)
            if DEBUG == True:
                if idx % 100 == 1:
                    #fout.write("%f, %f, %f\n" % (self.race_score[1000][0], self.race_score[1000][10], self.race_score[1000][20]))
                    fout.write("%f, %f, %f, %f\n" % (self.hr_days[1000], self.hr_history_total[1000][4], self.jk_history_total[1000][4], self.tr_history_total[1000][4]))
        for i in range(21):
            self.race_score[0][i] = (self.race_score[900][i] + self.race_score[1000][i] + self.race_score[1200][i] + self.race_score[1300][i] + self.race_score[1400][i] + self.race_score[1600][i]) / 6



def update_md(fname):
    data = pd.read_csv(fname)
    md = mean_data()
    md.update_data(data)
    joblib.dump(md, fname.replace('.csv', '_md.pkl'))


if __name__ == '__main__':
    DEBUG = True
    fname_csv = '../data/3_2007_2016.csv'
    update_md(fname_csv)

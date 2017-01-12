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
            {300:  [[707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707]],
             400:  [[321, 321, 321, 328, 327, 328, 328, 326, 328, 331, 331, 333, 328, 326, 321, 330, 327, 322, 331, 323, 327],
                    [321, 321, 321, 328, 327, 328, 328, 326, 328, 331, 331, 333, 328, 326, 321, 330, 327, 322, 331, 323, 327],
                    [321, 321, 321, 328, 327, 328, 328, 326, 328, 331, 331, 333, 328, 326, 321, 330, 327, 322, 331, 323, 327],
                    [321, 321, 321, 328, 327, 328, 328, 326, 328, 331, 331, 333, 328, 326, 321, 330, 327, 322, 331, 323, 327],
                    [321, 321, 321, 328, 327, 328, 328, 326, 328, 331, 331, 333, 328, 326, 321, 330, 327, 322, 331, 323, 327],
                    [321, 321, 321, 328, 327, 328, 328, 326, 328, 331, 331, 333, 328, 326, 321, 330, 327, 322, 331, 323, 327],
                    [321, 321, 321, 328, 327, 328, 328, 326, 328, 331, 331, 333, 328, 326, 321, 330, 327, 322, 331, 323, 327],
                    [321, 321, 321, 328, 327, 328, 328, 326, 328, 331, 331, 333, 328, 326, 321, 330, 327, 322, 331, 323, 327],
                    [321, 321, 321, 328, 327, 328, 328, 326, 328, 331, 331, 333, 328, 326, 321, 330, 327, 322, 331, 323, 327],
                    [321, 321, 321, 328, 327, 328, 328, 326, 328, 331, 331, 333, 328, 326, 321, 330, 327, 322, 331, 323, 327],
                    [321, 321, 321, 328, 327, 328, 328, 326, 328, 331, 331, 333, 328, 326, 321, 330, 327, 322, 331, 323, 327],
                    [321, 321, 321, 328, 327, 328, 328, 326, 328, 331, 331, 333, 328, 326, 321, 330, 327, 322, 331, 323, 327]],
             800:  [[707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707],
                    [707, 707, 707, 706, 703, 707, 709, 707, 709, 709, 702, 710, 706, 707, 713, 709, 712, 700, 712, 710, 707]],
             900:  [[703, 703, 703, 708, 705, 716, 708, 716, 718, 716, 723, 711, 717, 710, 712, 709, 702, 701, 718, 711, 712],
                    [703, 703, 703, 708, 705, 716, 708, 716, 718, 716, 723, 711, 717, 710, 712, 709, 702, 701, 718, 711, 712],
                    [703, 703, 703, 708, 705, 716, 708, 716, 718, 716, 723, 711, 717, 710, 712, 709, 702, 701, 718, 711, 712],
                    [703, 703, 703, 708, 705, 716, 708, 716, 718, 716, 723, 711, 717, 710, 712, 709, 702, 701, 718, 711, 712],
                    [703, 703, 703, 708, 705, 716, 708, 716, 718, 716, 723, 711, 717, 710, 712, 709, 702, 701, 718, 711, 712],
                    [703, 703, 703, 708, 705, 716, 708, 716, 718, 716, 723, 711, 717, 710, 712, 709, 702, 701, 718, 711, 712],
                    [703, 703, 703, 708, 705, 716, 708, 716, 718, 716, 723, 711, 717, 710, 712, 709, 702, 701, 718, 711, 712],
                    [703, 703, 703, 708, 705, 716, 708, 716, 718, 716, 723, 711, 717, 710, 712, 709, 702, 701, 718, 711, 712],
                    [703, 703, 703, 708, 705, 716, 708, 716, 718, 716, 723, 711, 717, 710, 712, 709, 702, 701, 718, 711, 712],
                    [703, 703, 703, 708, 705, 716, 708, 716, 718, 716, 723, 711, 717, 710, 712, 709, 702, 701, 718, 711, 712],
                    [703, 703, 703, 708, 705, 716, 708, 716, 718, 716, 723, 711, 717, 710, 712, 709, 702, 701, 718, 711, 712],
                    [703, 703, 703, 708, 705, 716, 708, 716, 718, 716, 723, 711, 717, 710, 712, 709, 702, 701, 718, 711, 712]],
             1000: [[753, 753, 753, 756, 754, 757, 764, 751, 762, 756, 763, 753, 758, 759, 759, 764, 758, 752, 747, 761, 757],
                    [753, 753, 753, 756, 754, 757, 764, 751, 762, 756, 763, 753, 758, 759, 759, 764, 758, 752, 747, 761, 757],
                    [753, 753, 753, 756, 754, 757, 764, 751, 762, 756, 763, 753, 758, 759, 759, 764, 758, 752, 747, 761, 757],
                    [753, 753, 753, 756, 754, 757, 764, 751, 762, 756, 763, 753, 758, 759, 759, 764, 758, 752, 747, 761, 757],
                    [753, 753, 753, 756, 754, 757, 764, 751, 762, 756, 763, 753, 758, 759, 759, 764, 758, 752, 747, 761, 757],
                    [753, 753, 753, 756, 754, 757, 764, 751, 762, 756, 763, 753, 758, 759, 759, 764, 758, 752, 747, 761, 757],
                    [753, 753, 753, 756, 754, 757, 764, 751, 762, 756, 763, 753, 758, 759, 759, 764, 758, 752, 747, 761, 757],
                    [753, 753, 753, 756, 754, 757, 764, 751, 762, 756, 763, 753, 758, 759, 759, 764, 758, 752, 747, 761, 757],
                    [753, 753, 753, 756, 754, 757, 764, 751, 762, 756, 763, 753, 758, 759, 759, 764, 758, 752, 747, 761, 757],
                    [753, 753, 753, 756, 754, 757, 764, 751, 762, 756, 763, 753, 758, 759, 759, 764, 758, 752, 747, 761, 757],
                    [753, 753, 753, 756, 754, 757, 764, 751, 762, 756, 763, 753, 758, 759, 759, 764, 758, 752, 747, 761, 757],
                    [753, 753, 753, 756, 754, 757, 764, 751, 762, 756, 763, 753, 758, 759, 759, 764, 758, 752, 747, 761, 757]],
             1200: [[879, 879, 879, 896, 888, 896, 893, 888, 887, 898, 885, 905, 908, 904, 901, 896, 887, 888, 895, 899, 895],
                    [879, 879, 879, 896, 888, 896, 893, 888, 887, 898, 885, 905, 908, 904, 901, 896, 887, 888, 895, 899, 895],
                    [879, 879, 879, 896, 888, 896, 893, 888, 887, 898, 885, 905, 908, 904, 901, 896, 887, 888, 895, 899, 895],
                    [879, 879, 879, 896, 888, 896, 893, 888, 887, 898, 885, 905, 908, 904, 901, 896, 887, 888, 895, 899, 895],
                    [879, 879, 879, 896, 888, 896, 893, 888, 887, 898, 885, 905, 908, 904, 901, 896, 887, 888, 895, 899, 895],
                    [879, 879, 879, 896, 888, 896, 893, 888, 887, 898, 885, 905, 908, 904, 901, 896, 887, 888, 895, 899, 895],
                    [879, 879, 879, 896, 888, 896, 893, 888, 887, 898, 885, 905, 908, 904, 901, 896, 887, 888, 895, 899, 895],
                    [879, 879, 879, 896, 888, 896, 893, 888, 887, 898, 885, 905, 908, 904, 901, 896, 887, 888, 895, 899, 895],
                    [879, 879, 879, 896, 888, 896, 893, 888, 887, 898, 885, 905, 908, 904, 901, 896, 887, 888, 895, 899, 895],
                    [879, 879, 879, 896, 888, 896, 893, 888, 887, 898, 885, 905, 908, 904, 901, 896, 887, 888, 895, 899, 895],
                    [879, 879, 879, 896, 888, 896, 893, 888, 887, 898, 885, 905, 908, 904, 901, 896, 887, 888, 895, 899, 895],
                    [879, 879, 879, 896, 888, 896, 893, 888, 887, 898, 885, 905, 908, 904, 901, 896, 887, 888, 895, 899, 895]],
             0:    [[673, 673, 673, 679, 675, 681, 680, 678, 681, 682, 681, 683, 683, 681, 681, 681, 677, 673, 681, 681, 680],
                    [673, 673, 673, 679, 675, 681, 680, 678, 681, 682, 681, 683, 683, 681, 681, 681, 677, 673, 681, 681, 680],
                    [673, 673, 673, 679, 675, 681, 680, 678, 681, 682, 681, 683, 683, 681, 681, 681, 677, 673, 681, 681, 680],
                    [673, 673, 673, 679, 675, 681, 680, 678, 681, 682, 681, 683, 683, 681, 681, 681, 677, 673, 681, 681, 680],
                    [673, 673, 673, 679, 675, 681, 680, 678, 681, 682, 681, 683, 683, 681, 681, 681, 677, 673, 681, 681, 680],
                    [673, 673, 673, 679, 675, 681, 680, 678, 681, 682, 681, 683, 683, 681, 681, 681, 677, 673, 681, 681, 680],
                    [673, 673, 673, 679, 675, 681, 680, 678, 681, 682, 681, 683, 683, 681, 681, 681, 677, 673, 681, 681, 680],
                    [673, 673, 673, 679, 675, 681, 680, 678, 681, 682, 681, 683, 683, 681, 681, 681, 677, 673, 681, 681, 680],
                    [673, 673, 673, 679, 675, 681, 680, 678, 681, 682, 681, 683, 683, 681, 681, 681, 677, 673, 681, 681, 680],
                    [673, 673, 673, 679, 675, 681, 680, 678, 681, 682, 681, 683, 683, 681, 681, 681, 677, 673, 681, 681, 680],
                    [673, 673, 673, 679, 675, 681, 680, 678, 681, 682, 681, 683, 683, 681, 681, 681, 677, 673, 681, 681, 680],
                    [673, 673, 673, 679, 675, 681, 680, 678, 681, 682, 681, 683, 683, 681, 681, 681, 677, 673, 681, 681, 680]]
             }
        self.course_record = [707, 327, 707, 712, 757, 895, 680]

        self.dist_rec = {400: [1.857689365, 2.068920101, 6.831353153, 327, 327, 327],
                         800: [0.663109756, 2.794664634, 7.339786585, 707, 707, 707],
                         900: [2.853055535, 5.294299758, 11.2948243, 712, 712, 712],
                         1000: [2.410946811, 4.767192673, 11.25186493, 757, 757, 757],
                         1110: [3.292790411, 8.250559669, 17.02678477, 932, 932, 932],
                         1200: [2.351360721, 6.505806899, 14.31563529, 895, 895, 895],
                         1400: [3.537566062, 9.716197686, 21.74241537, 1052, 1052, 1052],
                         1610: [3.113955119, 9.534730014, 20.65406732, 1213, 1213, 1213],
                         1700: [4.688991532, 10.26871953, 23.47564793, 1267, 1267, 1267],
                         1800: [1.33877551, 6.65755102, 20.69591837, 1342, 1342, 1342]}
        self.hr_days = {400:  1169,
                        800:  1215,
                        900:  1174,
                        1000: 1252,
                        1110: 1926,
                        1200: 1339,
                        1400: 1398,
                        1610: 1544,
                        1700: 1744,
                        1800: 1763}
        self.hr_history_total = {400:  [7.33438, 0.42241, 0.64418, 5.92169, 8.70992],
                                 800:  [10.80972, 0.81085, 1.10555, 10.82229, 11.24310],
                                 900:  [9.05387, 1.07201, 1.03966, 7.08036, 5.48896],
                                 1000: [12.24618, 1.70048, 1.62501, 15.10616, 14.97908],
                                 1110: [46.26471, 10.74131, 7.39572, 26.31016, 15.72861],
                                 1200: [16.25300, 2.52812, 2.11706, 18.52602, 13.65891],
                                 1400: [17.82332, 2.59156, 2.38734, 19.26692, 14.49546],
                                 1610: [23.79572, 4.10619, 3.40118, 20.88569, 15.04388],
                                 1700: [31.86801, 5.96329, 4.76049, 21.49476, 15.54283],
                                 1800: [32.32210, 6.53745, 4.86891, 23.30150, 15.51685]
                                 }

        self.hr_history_year = {400:  [5.73621, 0.30523, 0.50443, 5.79080, 8.63018], 
                                800:  [7.81807, 0.60255, 0.83205, 10.55883, 11.11015], 
                                900:  [5.63123, 0.61145, 0.58905, 6.33222, 5.11320], 
                                1000: [8.44479, 0.91365, 1.02722, 13.84033, 14.51921], 
                                1110: [14.60428, 2.28342, 2.07019, 15.96725, 13.22193], 
                                1200: [11.11268, 1.49380, 1.39886, 17.43527, 13.35062], 
                                1400: [12.31644, 1.86581, 1.67917, 18.33724, 13.94316], 
                                1610: [13.07559, 2.21792, 1.92257, 18.59292, 14.51254], 
                                1700: [13.95455, 2.39073, 2.14598, 17.66871, 14.81556], 
                                1800: [13.98127, 2.55805, 2.16105, 18.71723, 15.20974],
                                }

        self.jk_history_total = {400:  [2848.35382, 311.91540, 307.83967, 9.52529, 9.54330],
                                 800:  [2869.34453, 320.21148, 313.54729, 9.75802, 9.74442],
                                 900:  [2695.84036, 301.50290, 293.52082, 9.76437, 9.76196],
                                 1000: [2740.92500, 312.05387, 301.12472, 10.01761, 9.86587],
                                 1110: [3107.96457, 362.73930, 345.61364, 10.43650, 9.98997],
                                 1200: [2828.18646, 324.47188, 310.98627, 10.17045, 9.96244],
                                 1400: [2788.45825, 322.37328, 307.26282, 10.22531, 9.88544],
                                 1610: [2863.59440, 337.94285, 317.66704, 10.55752, 10.06674],
                                 1700: [3079.46241, 364.83042, 343.35052, 10.73689, 10.14248],
                                 1800: [3048.09176, 370.79213, 341.32022, 10.97378, 10.19288]
                                 }
        self.jk_history_year = {400:  [275.12118, 30.40154, 30.04687, 9.53987, 9.78851],
                                800:  [288.55170, 32.91180, 32.01239, 9.94746, 10.05067],
                                900:  [280.84435, 32.05562, 30.98186, 10.04305, 10.05043],
                                1000: [288.16382, 33.66816, 32.28654, 10.34420, 10.22373],
                                1110: [322.78877, 39.49131, 36.91644, 11.02674, 10.57019],
                                1200: [296.28694, 35.27512, 33.35214, 10.60248, 10.36930],
                                1400: [294.27219, 34.91005, 33.13478, 10.60094, 10.33372],
                                1610: [305.36763, 37.30236, 34.71829, 11.07780, 10.52323],
                                1700: [314.03584, 38.65297, 36.22727, 11.21416, 10.68182],
                                1800: [320.12734, 40.22472, 36.79588, 11.50936, 10.71348]
                                }

        self.tr_history_total = {400:  [5816.77994, 612.64904, 637.01629, 10.27322, 10.51843],
                                 800:  [5588.66288, 592.19572, 613.24085, 10.43732, 10.57684],
                                 900:  [5333.55217, 569.13855, 586.51683, 10.39352, 10.51754],
                                 1000: [5302.03779, 562.93493, 582.34730, 10.34130, 10.49934],
                                 1110: [5192.40374, 583.00869, 580.40107, 11.04078, 10.84358],
                                 1200: [5370.06501, 571.74337, 590.56015, 10.45148, 10.54833],
                                 1400: [5531.69470, 584.90185, 605.63961, 10.37181, 10.47846],
                                 1610: [5402.42588, 569.01770, 588.08776, 10.41925, 10.41962],
                                 1700: [5420.35402, 568.92483, 589.02273, 10.43444, 10.45017],
                                 1800: [5340.65169, 560.52434, 579.23034, 10.50375, 10.45318]
                                 }
        self.tr_history_year = {400:  [395.65419, 41.43555, 42.40869, 9.93484, 10.15319],
                                800:  [400.54832, 43.18456, 43.56784, 10.25061, 10.33327],
                                900:  [400.63598, 43.07998, 43.07468, 10.21405, 10.20886],
                                1000: [404.50313, 43.88700, 43.89751, 10.29117, 10.29905],
                                1110: [406.98663, 48.21257, 45.76270, 11.25802, 10.70588],
                                1200: [405.38551, 44.81544, 43.97369, 10.49037, 10.30067],
                                1400: [405.57515, 44.41342, 43.83651, 10.41518, 10.25139],
                                1610: [403.79757, 44.78872, 43.86578, 10.55826, 10.30937],
                                1700: [406.86451, 45.28759, 44.94231, 10.60402, 10.51224],
                                1800: [406.47753, 45.95131, 45.28277, 10.80524, 10.57865]
                                }
        self.race_detail = { 300: [190, 200, 540],
                             400: [190, 200, 540],
                             800: [190, 200, 540],
                             900: [190, 200, 540],
                            1000: [190, 200, 540],
                            1200: [190, 200, 540],
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

    def update_race_score_qual(self, month, humidity, record):
        if humidity > 20:
            humidity = 20
        humidity -= 1
        try:
            self.race_score[300][month][humidity] += self.lr * (record - self.race_score[300][month][humidity])
            self.race_score[300][month][20] = np.mean(self.race_score[300][month])
        except KeyError:
            return 0

    def update_race_score(self, course, month, humidity, data):
        humidity = min(humidity, 20) - 1
        try:
            self.race_score[course][month][humidity] += self.lr * (data['rctime'] - self.race_score[course][month][humidity])
            self.race_score[course][month][20] = np.mean(self.race_score[course][month][:20])
        except KeyError:
            return 0

    def update_race_detail(self, course, data):
        try:
            self.race_detail[course][0] += self.lr * (data['s1f'] - self.race_score[course][0])
            self.race_detail[course][1] += self.lr * (data['g1f'] - self.race_score[course][1])
            self.race_detail[course][2] += self.lr * (data['g3f'] - self.race_score[course][2])
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
            self.update_race_score(int(row['course']), int(row['month'])-1, int(row['humidity']), row)
            self.update_race_detail(int(row['course']), row)
            if DEBUG == True:
                if idx % 100 == 1:
                    #fout.write("%f, %f, %f\n" % (self.race_score[400][0], self.race_score[400][10], self.race_score[400][20]))
                    fout.write("%f, %f, %f, %f\n" % (self.hr_days[400], self.hr_history_total[400][4], self.jk_history_total[400][4], self.tr_history_total[400][4]))
        for m in range(12):
            for i in range(21):
                self.race_score[0][m][i] = (self.race_score[300][m][i] + self.race_score[400][m][i] + self.race_score[800][m][i] + self.race_score[900][m][i] + self.race_score[1000][m][i] + self.race_score[1200][m][i]) / 6



def update_md(fname):
    data = pd.read_csv(fname)
    md = mean_data()
    md.update_data(data)
    joblib.dump(md, fname.replace('.csv', '_md.pkl'))


if __name__ == '__main__':
    DEBUG = True
    fname_csv = '../data/2_2007_2016.csv'
    update_md(fname_csv)

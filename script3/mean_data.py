# -*- coding:utf-8 -*-

import sys
if sys.version_info >= (3, 0):
    from urllib.request import urlopen
else:
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
        self.lr = 0.01

        self.race_score = \
            {900:  [[629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633]],
             1000: [[629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633],
                    [629, 641, 640, 639, 638, 640, 638, 640, 640, 639, 640, 640, 639, 635, 641, 638, 637, 635, 635, 633, 633]],
             1200: [[768, 771, 778, 776, 777, 775, 773, 775, 778, 778, 779, 776, 777, 774, 776, 769, 772, 771, 772, 769, 769],
                    [768, 771, 778, 776, 777, 775, 773, 775, 778, 778, 779, 776, 777, 774, 776, 769, 772, 771, 772, 769, 769],
                    [768, 771, 778, 776, 777, 775, 773, 775, 778, 778, 779, 776, 777, 774, 776, 769, 772, 771, 772, 769, 769],
                    [768, 771, 778, 776, 777, 775, 773, 775, 778, 778, 779, 776, 777, 774, 776, 769, 772, 771, 772, 769, 769],
                    [768, 771, 778, 776, 777, 775, 773, 775, 778, 778, 779, 776, 777, 774, 776, 769, 772, 771, 772, 769, 769],
                    [768, 771, 778, 776, 777, 775, 773, 775, 778, 778, 779, 776, 777, 774, 776, 769, 772, 771, 772, 769, 769],
                    [768, 771, 778, 776, 777, 775, 773, 775, 778, 778, 779, 776, 777, 774, 776, 769, 772, 771, 772, 769, 769],
                    [768, 771, 778, 776, 777, 775, 773, 775, 778, 778, 779, 776, 777, 774, 776, 769, 772, 771, 772, 769, 769],
                    [768, 771, 778, 776, 777, 775, 773, 775, 778, 778, 779, 776, 777, 774, 776, 769, 772, 771, 772, 769, 769],
                    [768, 771, 778, 776, 777, 775, 773, 775, 778, 778, 779, 776, 777, 774, 776, 769, 772, 771, 772, 769, 769],
                    [768, 771, 778, 776, 777, 775, 773, 775, 778, 778, 779, 776, 777, 774, 776, 769, 772, 771, 772, 769, 769],
                    [768, 771, 778, 776, 777, 775, 773, 775, 778, 778, 779, 776, 777, 774, 776, 769, 772, 771, 772, 769, 769]],
             1300: [[837, 836, 847, 844, 843, 843, 841, 844, 847, 848, 845, 846, 843, 840, 845, 836, 840, 841, 840, 836, 836],
                    [837, 836, 847, 844, 843, 843, 841, 844, 847, 848, 845, 846, 843, 840, 845, 836, 840, 841, 840, 836, 836],
                    [837, 836, 847, 844, 843, 843, 841, 844, 847, 848, 845, 846, 843, 840, 845, 836, 840, 841, 840, 836, 836],
                    [837, 836, 847, 844, 843, 843, 841, 844, 847, 848, 845, 846, 843, 840, 845, 836, 840, 841, 840, 836, 836],
                    [837, 836, 847, 844, 843, 843, 841, 844, 847, 848, 845, 846, 843, 840, 845, 836, 840, 841, 840, 836, 836],
                    [837, 836, 847, 844, 843, 843, 841, 844, 847, 848, 845, 846, 843, 840, 845, 836, 840, 841, 840, 836, 836],
                    [837, 836, 847, 844, 843, 843, 841, 844, 847, 848, 845, 846, 843, 840, 845, 836, 840, 841, 840, 836, 836],
                    [837, 836, 847, 844, 843, 843, 841, 844, 847, 848, 845, 846, 843, 840, 845, 836, 840, 841, 840, 836, 836],
                    [837, 836, 847, 844, 843, 843, 841, 844, 847, 848, 845, 846, 843, 840, 845, 836, 840, 841, 840, 836, 836],
                    [837, 836, 847, 844, 843, 843, 841, 844, 847, 848, 845, 846, 843, 840, 845, 836, 840, 841, 840, 836, 836],
                    [837, 836, 847, 844, 843, 843, 841, 844, 847, 848, 845, 846, 843, 840, 845, 836, 840, 841, 840, 836, 836],
                    [837, 836, 847, 844, 843, 843, 841, 844, 847, 848, 845, 846, 843, 840, 845, 836, 840, 841, 840, 836, 836]],
             1400: [[897, 899, 906, 904, 905, 904, 903, 903, 908, 910, 908, 903, 901, 904, 900, 897, 901, 897, 899, 894, 894],
                    [897, 899, 906, 904, 905, 904, 903, 903, 908, 910, 908, 903, 901, 904, 900, 897, 901, 897, 899, 894, 894],
                    [897, 899, 906, 904, 905, 904, 903, 903, 908, 910, 908, 903, 901, 904, 900, 897, 901, 897, 899, 894, 894],
                    [897, 899, 906, 904, 905, 904, 903, 903, 908, 910, 908, 903, 901, 904, 900, 897, 901, 897, 899, 894, 894],
                    [897, 899, 906, 904, 905, 904, 903, 903, 908, 910, 908, 903, 901, 904, 900, 897, 901, 897, 899, 894, 894],
                    [897, 899, 906, 904, 905, 904, 903, 903, 908, 910, 908, 903, 901, 904, 900, 897, 901, 897, 899, 894, 894],
                    [897, 899, 906, 904, 905, 904, 903, 903, 908, 910, 908, 903, 901, 904, 900, 897, 901, 897, 899, 894, 894],
                    [897, 899, 906, 904, 905, 904, 903, 903, 908, 910, 908, 903, 901, 904, 900, 897, 901, 897, 899, 894, 894],
                    [897, 899, 906, 904, 905, 904, 903, 903, 908, 910, 908, 903, 901, 904, 900, 897, 901, 897, 899, 894, 894],
                    [897, 899, 906, 904, 905, 904, 903, 903, 908, 910, 908, 903, 901, 904, 900, 897, 901, 897, 899, 894, 894],
                    [897, 899, 906, 904, 905, 904, 903, 903, 908, 910, 908, 903, 901, 904, 900, 897, 901, 897, 899, 894, 894],
                    [897, 899, 906, 904, 905, 904, 903, 903, 908, 910, 908, 903, 901, 904, 900, 897, 901, 897, 899, 894, 894]],
             1700: [[1118, 1137, 1146, 1144, 1145, 1145, 1142, 1145, 1144, 1148, 1150, 1144, 1143, 1144, 1133, 1139, 1136, 1131, 1130, 1133, 1133],
                    [1118, 1137, 1146, 1144, 1145, 1145, 1142, 1145, 1144, 1148, 1150, 1144, 1143, 1144, 1133, 1139, 1136, 1131, 1130, 1133, 1133],
                    [1118, 1137, 1146, 1144, 1145, 1145, 1142, 1145, 1144, 1148, 1150, 1144, 1143, 1144, 1133, 1139, 1136, 1131, 1130, 1133, 1133],
                    [1118, 1137, 1146, 1144, 1145, 1145, 1142, 1145, 1144, 1148, 1150, 1144, 1143, 1144, 1133, 1139, 1136, 1131, 1130, 1133, 1133],
                    [1118, 1137, 1146, 1144, 1145, 1145, 1142, 1145, 1144, 1148, 1150, 1144, 1143, 1144, 1133, 1139, 1136, 1131, 1130, 1133, 1133],
                    [1118, 1137, 1146, 1144, 1145, 1145, 1142, 1145, 1144, 1148, 1150, 1144, 1143, 1144, 1133, 1139, 1136, 1131, 1130, 1133, 1133],
                    [1118, 1137, 1146, 1144, 1145, 1145, 1142, 1145, 1144, 1148, 1150, 1144, 1143, 1144, 1133, 1139, 1136, 1131, 1130, 1133, 1133],
                    [1118, 1137, 1146, 1144, 1145, 1145, 1142, 1145, 1144, 1148, 1150, 1144, 1143, 1144, 1133, 1139, 1136, 1131, 1130, 1133, 1133],
                    [1118, 1137, 1146, 1144, 1145, 1145, 1142, 1145, 1144, 1148, 1150, 1144, 1143, 1144, 1133, 1139, 1136, 1131, 1130, 1133, 1133],
                    [1118, 1137, 1146, 1144, 1145, 1145, 1142, 1145, 1144, 1148, 1150, 1144, 1143, 1144, 1133, 1139, 1136, 1131, 1130, 1133, 1133],
                    [1118, 1137, 1146, 1144, 1145, 1145, 1142, 1145, 1144, 1148, 1150, 1144, 1143, 1144, 1133, 1139, 1136, 1131, 1130, 1133, 1133],
                    [1118, 1137, 1146, 1144, 1145, 1145, 1142, 1145, 1144, 1148, 1150, 1144, 1143, 1144, 1133, 1139, 1136, 1131, 1130, 1133, 1133]],
             0: [[813, 821, 826, 824, 824, 825, 823, 825, 826, 827, 827, 825, 824, 822, 823, 820, 821, 818, 819, 816, 816],
                 [813, 821, 826, 824, 824, 825, 823, 825, 826, 827, 827, 825, 824, 822, 823, 820, 821, 818, 819, 816, 816],
                 [813, 821, 826, 824, 824, 825, 823, 825, 826, 827, 827, 825, 824, 822, 823, 820, 821, 818, 819, 816, 816],
                 [813, 821, 826, 824, 824, 825, 823, 825, 826, 827, 827, 825, 824, 822, 823, 820, 821, 818, 819, 816, 816],
                 [813, 821, 826, 824, 824, 825, 823, 825, 826, 827, 827, 825, 824, 822, 823, 820, 821, 818, 819, 816, 816],
                 [813, 821, 826, 824, 824, 825, 823, 825, 826, 827, 827, 825, 824, 822, 823, 820, 821, 818, 819, 816, 816],
                 [813, 821, 826, 824, 824, 825, 823, 825, 826, 827, 827, 825, 824, 822, 823, 820, 821, 818, 819, 816, 816],
                 [813, 821, 826, 824, 824, 825, 823, 825, 826, 827, 827, 825, 824, 822, 823, 820, 821, 818, 819, 816, 816],
                 [813, 821, 826, 824, 824, 825, 823, 825, 826, 827, 827, 825, 824, 822, 823, 820, 821, 818, 819, 816, 816],
                 [813, 821, 826, 824, 824, 825, 823, 825, 826, 827, 827, 825, 824, 822, 823, 820, 821, 818, 819, 816, 816],
                 [813, 821, 826, 824, 824, 825, 823, 825, 826, 827, 827, 825, 824, 822, 823, 820, 821, 818, 819, 816, 816],
                 [813, 821, 826, 824, 824, 825, 823, 825, 826, 827, 827, 825, 824, 822, 823, 820, 821, 818, 819, 816, 816]]
             }

        self.dist_rec = {1000: [1.857689365, 2.068920101, 6.831353153, 636, 642, 648],
                         1100: [0.663109756, 2.794664634, 7.339786585, 702, 705, 708],
                         1200: [2.853055535, 5.294299758, 11.2948243, 771, 779, 789],
                         1300: [2.410946811, 4.767192673, 11.25186493, 837, 845, 853],
                         1400: [3.292790411, 8.250559669, 17.02678477, 894, 904, 915],
                         1700: [2.351360721, 6.505806899, 14.31563529, 1133, 1143, 1154],
                         1800: [3.537566062, 9.716197686, 21.74241537, 1193, 1206, 1221],
                         1900: [3.113955119, 9.534730014, 20.65406732, 1256, 1270, 1283],
                         2000: [4.688991532, 10.26871953, 23.47564793, 1308, 1329, 1347],
                         2300: [1.33877551, 6.65755102, 20.69591837, 1497, 1512, 1529]}
        self.hr_days = {1000: 1107.149415,
                        1100: 1239.905488,
                        1200: 1253.701232,
                        1300: 1356.83166,
                        1400: 1492.907941,
                        1700: 1544.546889,
                        1800: 1672.514569,
                        1900: 1845.470196,
                        2000: 1948.556069,
                        2300: 1849.616327}
        self.hr_history_total = {1000: [3.732429774, 0.403869275, 0.518471964, 4.677833643, 7.254781943], #nt, nt1, nt2, t1, t2
                                 1100: [7.045731707, 0.442073171, 0.731707317, 7.541158537, 8.87195122],
                                 1200: [7.52446629, 0.512141442, 0.619396963, 6.923189788, 7.410938302],
                                 1300: [10.38981461, 0.596501876, 0.803189141, 8.017932024, 8.516442286],
                                 1400: [14.16155473, 1.410717433, 1.474528468, 12.59232329, 11.09399788],
                                 1700: [16.04862194, 1.605217542, 1.732362628, 12.95891836, 11.83246663],
                                 1800: [19.22482502, 2.535637766, 2.390515641, 15.69618626, 13.01564062],
                                 1900: [23.21704067, 3.929347826, 3.241234222, 19.35466339, 14.39235624],
                                 2000: [25.39132666, 5.339235309, 3.922504491, 23.27046446, 15.53938927],
                                 2300: [22.53877551, 5.706122449, 3.404081633, 24.63265306, 14.05714286]
                                 }

        self.hr_history_year = {1000: [3.148212919, 0.37129741, 0.478740846, 4.043228768, 6.975352498], #ny, ny1, ny2, y1, y2
                                1100: [5.506097561, 0.405487805, 0.62652439, 7.321646341, 8.745426829],
                                1200: [5.67827012, 0.369928839, 0.461741618, 6.531325655, 7.19672071],
                                1300: [7.475446921, 0.460770249, 0.608585301, 7.827962922, 8.41083646],
                                1400: [8.573329808, 0.827736647, 0.872862683, 11.54605147, 10.47474881],
                                1700: [9.296151846, 1.000433351, 1.037268157, 12.29693188, 11.37354828],
                                1800: [9.282316812, 1.183116698, 1.117697472, 13.65840594, 11.87644622],
                                1900: [9.052945302, 1.329593268, 1.130960729, 15.10045582, 12.14954418],
                                2000: [8.791121375, 1.475750577, 1.149089043, 16.90890429, 12.71901463],
                                2300: [8.497959184, 1.67755102, 1, 20.34285714, 11.53877551]
                                }

        self.jk_history_total = {1000: [2285.378839, 244.1428571, 234.3842496, 7.985572194, 7.999781397],
                                 1100: [2330.042683, 263.6310976, 246.9801829, 8.19054878, 8.402439024],
                                 1200: [2334.656005, 252.6178564, 241.5765168, 8.017020028, 8.022999046],
                                 1300: [2253.860351, 241.6955418, 231.2209777, 7.911388215, 7.90427058],
                                 1400: [2338.3581, 257.5880046, 245.8177331, 8.132425524, 8.122554204],
                                 1700: [2368.441498, 260.9362108, 249.9377708, 8.23400936, 8.185214075],
                                 1800: [2468.537138, 278.5812741, 264.2802457, 8.530567062, 8.364662191],
                                 1900: [2445.940393, 273.7023142, 260.8053997, 8.500175316, 8.328541374],
                                 2000: [2614.934308, 298.6325378, 283.659225, 8.810110341, 8.613805491],
                                 2300: [2651.453061, 309.2285714, 290.877551, 9.265306122, 9.040816327]
                                 }
        self.jk_history_year = {1000: [288.5567275, 29.80353044, 28.39069844, 7.939501585, 8.028090502],
                                1100: [292.3125, 31.37957317, 29.14481707, 8.221036585, 8.544207317],
                                1200: [289.8175482, 30.08785122, 28.63113491, 7.924546988, 8.051646981],
                                1300: [286.0542927, 29.39715295, 28.00005518, 7.803796072, 7.977598764],
                                1400: [294.0408514, 31.04230566, 29.36175745, 8.185439803, 8.188304248],
                                1700: [301.5441151, 32.08771018, 30.44869128, 8.404489513, 8.356387589],
                                1800: [310.972004, 33.90965576, 31.80995572, 8.711255535, 8.564847879],
                                1900: [314.6597125, 34.39586255, 32.30399719, 8.768232819, 8.564165498],
                                2000: [324.3556582, 36.52501925, 34.29509879, 9.092635361, 8.859122402],
                                2300: [326.0897959, 37.14285714, 34.94285714, 9.595918367, 9.33877551]
                                }

        self.tr_history_total = {1000: [3382.346541, 317.6670674, 321.9784676, 8.797846759, 8.711334572],
                                 1100: [3474.550305, 313.5182927, 325.7606707, 8.384146341, 8.407012195],
                                 1200: [3443.97425, 323.4722324, 328.0756364, 8.829359548, 8.723791358],
                                 1300: [3441.879386, 323.1303796, 327.3898698, 8.776373869, 8.698907526],
                                 1400: [3525.777543, 333.2326371, 336.9733827, 8.935219461, 8.795126036],
                                 1700: [3521.138066, 334.4570116, 337.6981279, 8.937597504, 8.823366268],
                                 1800: [3511.059349, 334.7203971, 337.5096415, 9.010784174, 8.824167976],
                                 1900: [3780.194074, 367.6982819, 367.4388149, 9.241409537, 8.980890603],
                                 2000: [3693.305876, 360.6217603, 358.8026687, 9.280728766, 9.010777521],
                                 2300: [3733.408163, 361.9673469, 359.9959184, 9.346938776, 9.012244898]
                                 }
        self.tr_history_year = {1000: [243.9066018, 21.4476992, 21.53601487, 8.194064925, 8.177341786],
                                1100: [252.7210366, 21.38719512, 22.36890244, 7.804878049, 8.150914634],
                                1200: [248.1472379, 21.60839263, 21.77716235, 8.104687844, 8.117782995],
                                1300: [252.9865372, 21.88071066, 22.18566542, 8.027863606, 8.112282057],
                                1400: [255.5646043, 22.80072272, 22.92935836, 8.358055702, 8.330468888],
                                1700: [256.866528, 23.2234356, 23.25931704, 8.470965505, 8.423383602],
                                1800: [260.4799314, 23.78560206, 23.68204542, 8.593986573, 8.442436795],
                                1900: [261.5943198, 24.68548387, 23.79610799, 8.90743338, 8.458800842],
                                2000: [261.2976649, 24.90325892, 24.15627406, 9.034642032, 8.608673338],
                                2300: [260.8489796, 25.04489796, 24.23673469, 9.204081633, 8.567346939]
                                }
        self.race_detail = { 900: [150, 150, 400],
                            1000: [150, 150, 400],
                            1200: [150, 150, 400],
                            1300: [150, 150, 400],
                            1400: [150, 150, 400],
                            1700: [150, 150, 400],
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
            self.race_score[900][month][humidity] += self.lr * (record - self.race_score[900][month][humidity])
            self.race_score[900][month][20] = np.mean(self.race_score[900][month])
        except KeyError:
            return 0

    def update_race_score(self, course, month, humidity, data):
        humidity = min(humidity, 20) - 1
        try:
            self.race_score[course][month][humidity] += self.lr * (data['rctime'] - self.race_score[course][month][humidity])
            self.race_score[course][month][20] = np.mean(self.race_score[course][month][:20])
        except KeyError:
            return 0


    def update_data(self, df):
        if DEBUG == True:
            if not os.path.exists('../log'):
                os.makedirs('../log')
            fout = open('../log/md.log', 'w')
        for idx, row in df.iterrows():
            self.update_hr(int(row['course']), row)
            self.update_jk(int(row['course']), row)
            self.update_tr(int(row['course']), row)
            self.update_dist_rec(int(row['course']), row)
            self.update_race_score(int(row['course']), int(row['month'])-1, int(row['humidity']), row)
            if DEBUG == True:
                if idx % 100 == 1:
                    #fout.write("%f, %f, %f\n" % (self.race_score[1000][0], self.race_score[1000][10], self.race_score[1000][20]))
                    fout.write("%f, %f, %f, %f\n" % (self.hr_days[1000], self.hr_history_total[1000][4], self.jk_history_total[1000][4], self.tr_history_total[1000][4]))
        for m in range(12):
            for i in range(21):
                self.race_score[0][m][i] = (self.race_score[900][m][i] + self.race_score[1000][m][i] + self.race_score[1200][m][i] + self.race_score[1300][m][i] + self.race_score[1400][m][i] + self.race_score[1700][m][i]) / 6



def update_md(fname):
    data = pd.read_csv(fname)
    md = mean_data()
    md.update_data(data)
    joblib.dump(md, fname.replace('.csv', '_md.pkl'))


if __name__ == '__main__':
    DEBUG = True
    fname_csv = '../data/1_2007_2016_v1.csv'
    update_md(fname_csv)

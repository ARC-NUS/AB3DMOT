
from __future__ import print_function
import os.path, copy, numpy as np, time, sys
import json
from datetime import datetime
from KFTracking import happyTracker
from wen_utils import readJson
from check_iou_jsons import check_iou_json
import glob
#

bag_dir = '/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/'
basedir_total = sorted(glob.glob(bag_dir + "set*/"))

for i in range(len(basedir_total)):
    basedir = basedir_total[i]
    print (basedir)
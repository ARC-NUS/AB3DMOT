from __future__ import print_function
import os.path, copy, numpy as np, time, sys
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from utils import load_list_from_folder, fileparts, mkdir_if_missing
from scipy.spatial import ConvexHull
from pyquaternion import Quaternion
from wen_utils import STATE_SIZE, MEAS_SIZE, MOTION_MODEL, get_CV_F, get_CA_F, camRadarFuse, readCamera, readRadar,readJson, readLidar
import json
from datetime import datetime


pathIBEO = '/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/set_1/ecu_obj_list/ecu_obj_list.json'

with open(pathIBEO, "r") as json_file:
    dataIB = json.load(json_file).get('ibeo_obj')

q1 = Quaternion(axis=[0, 0, 1], angle=0)
T1 = q1.transformation_matrix

for frame_name in range(0,1000):
#def readIBEO(det_IBEO, frame_name, T1):
    det_IBEO = dataIB[frame_name]['data']
    dets_IBEO_temp = np.zeros([1, 9])
    pos_IBEO = np.zeros([4, 1])
    k = 0
    additional_info_2_temp = np.zeros([1, 7])

    for j in range(len(det_IBEO)):
    #for j in range(3):
        obj_class = det_IBEO[j]['obj_class']
        width = float(det_IBEO[j]['obj_size']['x'])
        length = float(det_IBEO[j]['obj_size']['y'])
        #print (obj_class)

        if obj_class > 2 and obj_class != 8 and width != 0 and length != 0 :

            if obj_class == 3:
                obj_class_foloyolo = 0
            if obj_class == 9  or obj_class == 4:
                obj_class_foloyolo = 1
            if obj_class == 5:
                obj_class_foloyolo = 2
            if obj_class == 8:
                obj_class_foloyolo = 3
            # if obj_class == 5:
            #     obj_class_foloyolo = 4
            if obj_class == 6:
                obj_class_foloyolo = 5

            dets_IBEO_temp[0][0] = frame_name
            dets_IBEO_temp[0][1] = float(det_IBEO[j]['obj_center']['x']) /100 # x values in pixor , but y in world frame!!
            dets_IBEO_temp[0][2] = -float(det_IBEO[j]['obj_center']['y'])/100  # y values in pixor , but x in world frame
            dets_IBEO_temp[0][3] = 1
            dets_IBEO_temp[0][4] = det_IBEO[j]['yaw']
            dets_IBEO_temp[0][5] = float(det_IBEO[j]['obj_size']['x']) /100 # width is in the y direction for Louis
            dets_IBEO_temp[0][6] = float(det_IBEO[j]['obj_size']['y']) /100
            dets_IBEO_temp[0][7] = 1
            dets_IBEO_temp[0][8] = 4  # sensor type: 4 IBEO!!!

            pos_IBEO[0][0] = dets_IBEO_temp[0][1]
            pos_IBEO[1][0] = dets_IBEO_temp[0][2]
            pos_IBEO[2][0] = 0
            pos_IBEO[3][0] = 1
            T2 = np.matmul(T1, pos_IBEO)
            dets_IBEO_temp[0][1] = T2[0][0]
            dets_IBEO_temp[0][2] = T2[1][0]
            additional_info_2_temp[0, 1] = obj_class
            if k == 0:
                dets_IBEO = np.copy(dets_IBEO_temp)
                additional_info_2 = np.copy(additional_info_2_temp)

            else:
                dets_IBEO = np.vstack((dets_IBEO, dets_IBEO_temp))
                additional_info_2 = np.vstack((additional_info_2, additional_info_2_temp))
            k += 1
            #print ('k is = %d' %k)
            pos_IBEO = np.zeros([4, 1])

    print (dets_IBEO , additional_info_2)
#    return dets_IBEO , additional_info_2

#!/usr/bin/env python
# coding: utf-8

'''
 file for utils for linn
'''

import json
import numpy as np


STATE_SIZE = 14
MEAS_SIZE = 7

# set R based on pixor stats in json 
def px_stats_get_R(pixor_stats_json):
  with open(pixor_stats_json) as pixor_stats_file:
    data = json.load(pixor_stats_file, encoding="utf-8")
    var = data['var']
    # x y z theta l w h 
    R = np.identity(MEAS_SIZE) # KF measurement uncertainty/noise
    # tp, fp, fn, [x y w b theta]
    # pixor_outputs_tf_epoch_3_valloss_0.0093.json
    # 210 92 335 [0.03953874 0.00588307 0.02431999 0.39831919 0.00211127] precision@80%iou75 :  69.54%, recall:  54.69%
    R[0,0] = var[0] #x
    R[1,1] = var[1] #y
    R[2,2] = 10.**-5 #z
    R[3,3] = var[4] #theta
    R[4,4] = var[3] #l
    R[5,5] = var[2] #w
    R[6,6] = 10.**-5 #h
  return R
  
  
def px_stats_get_P_0(pixor_stats_json, p0_v=1000., factor=1.):
  with open(pixor_stats_json) as pixor_stats_file:
    data = json.load(pixor_stats_file, encoding="utf-8")
    var = data['var']
    # x y z theta l w h 
    P_0 = np.identity(STATE_SIZE) # KF measurement uncertainty/noise
    # tp, fp, fn, [x y w b theta]
    # pixor_outputs_tf_epoch_3_valloss_0.0093.json
    # 210 92 335 [0.03953874 0.00588307 0.02431999 0.39831919 0.00211127] precision@80%iou75 :  69.54%, recall:  54.69%
    P_0[0,0] = var[0] #x
    P_0[1,1] = var[1] #y
    P_0[2,2] = 10.**-5 #z
    P_0[3,3] = var[4] #theta
    P_0[4,4] = var[3] #l
    P_0[5,5] = var[2] #w
    P_0[6,6] = 10.**-5 #h
    P_0 = P_0*factor
    P_0[7,7] = p0_v # vx
    P_0[8,8] = p0_v # vy
    P_0[9,9] = p0_v # vz
  return P_0


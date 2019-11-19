#!/usr/bin/env python
# coding: utf-8

'''
 file for utils for linn
'''

import json
import numpy as np
from os import listdir, walk, pardir
from os.path import isfile, join


STATE_SIZE = 14
MEAS_SIZE = 7
MOTION_MODEL="CYRA"
#MOTION_MODEL="CA"
# MOTION_MODEL="CV"

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
    P_0[2,2] = 0. #z
    P_0[3,3] = var[4] #theta
    P_0[4,4] = var[3] #l
    P_0[5,5] = var[2] #w
    P_0[6,6] = 0. #h
    P_0 = P_0*factor
    P_0[7,7] = p0_v # vx
    P_0[8,8] = p0_v # vy
    P_0[9,9] = 0. # vz
    # print "P_O", P_0 
  return P_0

def get_CV_Q(q_v, delta_t):
  Q = np.zeros((STATE_SIZE, STATE_SIZE))
  Q[0,0]=delta_t**3*q_v/3.
  Q[1,1]=delta_t**3*q_v/3.
  Q[0,7]=delta_t**2*q_v/2.
  Q[1,8]=delta_t**2*q_v/2.
  Q[7,0]=delta_t**2*q_v/2.
  Q[8,1]=delta_t**2*q_v/2.
  Q[7,7]=delta_t*q_v
  Q[8,8]=delta_t*q_v
  # print "Q", Q
  return Q

def get_CV_F(delta_t):
  F = np.eye(STATE_SIZE)      
  F[0,7]=delta_t
  F[1,8]=delta_t
  F[2,9]=delta_t
  return F

def get_CA_Q(q_a, delta_t):
  Q = np.zeros((STATE_SIZE, STATE_SIZE))
  Q[0,0]=delta_t**5 * q_a / 20.
  Q[1,1]=delta_t**5 * q_a / 20.
  Q[0,7]=delta_t**4 * q_a / 8.
  Q[1,8]=delta_t**4 * q_a / 8.
  Q[0,10]=delta_t**3 * q_a / 6.
  Q[1,11]=delta_t**3 * q_a / 6.
  Q[7,0]=delta_t**4 * q_a / 8.
  Q[8,1]=delta_t**4 * q_a / 8.
  Q[10,0]=delta_t**3 * q_a / 6.
  Q[11,1]=delta_t**3 * q_a / 6.
  Q[7,7]=delta_t**3 * q_a / 3.
  Q[8,8]=delta_t**3 * q_a / 3.
  Q[7,10]=delta_t**2* q_a / 2.
  Q[8,11]=delta_t**2* q_a / 2.
  Q[7,10]=delta_t**2* q_a / 2.
  Q[8,11]=delta_t**2* q_a / 2.
  Q[10,10]=delta_t* q_a
  Q[11,11]=delta_t* q_a
  return Q

def get_CA_F(delta_t):
  F = np.eye(STATE_SIZE)     
  F[0,7] = delta_t
  F[1,8] = delta_t
  F[0,10] = delta_t **2. /2.
  F[1,11] = delta_t **2. /2.
  F[7,10] = delta_t
  F[8,11] = delta_t
  return F

def get_CYRA_Q(q_a, q_p, T):
  Q = np.zeros((STATE_SIZE, STATE_SIZE))
  Q[0,0]=T**5 * q_a / 20.
  Q[1,1]=T**5 * q_a / 20.
  Q[2,2]=T**2 * q_p / 3.
  Q[0,7]=T**4 * q_a / 8.
  Q[1,8]=T**4 * q_a / 8.
  Q[0,10]=T**2 * q_a / 6.
  Q[1,11]=T**2 * q_a / 6.
  Q[2,13]=T**2 * q_p / 2.
  Q[7,0]=T**4 * q_a / 8.
  Q[8,1]=T**4 * q_a / 8.
  Q[7,7]=T**2 * q_a / 3.
  Q[8,8]=T**2 * q_a / 3.
  Q[7,10]=T**2 * q_a / 2.
  Q[8,11]=T**2 * q_a / 2.
  Q[10,0]=T**2 * q_a / 6.
  Q[11,1]=T**2 * q_a / 6.
  Q[13,2]=T**2 * q_p / 2.
  Q[10,7]=T**2 * q_a / 2.
  Q[11,8]=T**2 * q_a / 2.
  Q[10,10]=T * q_a
  Q[11,11]=T * q_a
  Q[13,13]=T*q_p
  return Q

def get_CYRA_F(delta_t):
  F = np.eye(STATE_SIZE)     
  F[0,7] = delta_t
  F[1,8] = delta_t
  F[0,10] = delta_t **2. /2.
  F[1,11] = delta_t **2. /2.
  F[2,13] = delta_t
  F[7,10] = delta_t
  F[8,11] = delta_t
  return F

def get_dirs(parent_dir, label_dir_name):
  labels_paths_v = []
  high_set_v = []
  labels_jsons_v=[]
  
  for root, dirs, files in walk(parent_dir):
    for i, dire in enumerate(dirs): # identify where sets are using the "pixor_test" directories
      if dire == label_dir_name:
        labels_dir=join(root,dire)
        low_set_dir= join(labels_dir, pardir) # location of the log_low set

        set_name =  root.split('/')[-1] # set name

        # loc of log_high set
        high_set_dir=join(low_set_dir, pardir) 
        high_set_dir=join(high_set_dir, pardir) 
        high_set_dir=join(high_set_dir, "log_high") 
        high_set_dir=join(high_set_dir, set_name) 
        
        # check if set dir has pcds and fused_poses
        try:
          set_contents = listdir(high_set_dir)
          if "fused_pose" in set_contents and "pcds" in set_contents:
#             print "found set", high_set_dir
            high_set_v.append(high_set_dir)
            labels_paths_v.append(labels_dir)
            for file in listdir(labels_dir):
              if file.find("annotations.json") > 0:
                labels_jsons_v.append(labels_dir+'/'+file)
        except: #find the set folders of interest
            print "found labels but not pcds or fused pose json"
#   print "high hz sets paths: ", high_set_v
#   print "low hz labels paths:", labels_paths_v
#   print "low hz labels jsons:", labels_jsons_v

  return high_set_v, labels_jsons_v
  
  

  

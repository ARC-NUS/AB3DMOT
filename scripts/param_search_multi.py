#!/usr/bin/env python

# file used to get best params for tracker
import numpy as np
from os import listdir, walk, pardir
from os.path import isfile, join
import csv

from read_json_jic import get_tracker_json
from check_iou_jsons import check_iou_json
from coord_desc import coord_descent
from numba import prange, jit
import threading
from multiprocessing.pool import ThreadPool
import datetime
from yl_utils import STATE_SIZE, get_CA_Q,MOTION_MODEL

# params: R: x/y phi w/l v a phi_dot; Q: qv,qp; ha
def get_MOT_score(params,high_set_v, labels_paths_v,  max_age,min_hits, is_print = True):
  
  r_xy = params[0]
  r_ori = params[1]
  r_wl = params[2]
  r_v = params[3]
  r_a= params[3]
  r_ori_dot=params[5]
  q_a= params[6]
  q_ori_dot=params[7]
  ha = params[8]

  is_params_ok = True
  for p in params:
    if p <= 0.:
      is_params_ok = False
  if ha <= 0. or ha >= 1.:
    is_params_ok = False

  overall_MOTA = 0.
  overall_MOTP = 0.

  if not is_params_ok: # check validity of param and give really bad score if not valid
    print "warning: params for coord desc in get_MOT_score must be positive"
    return -np.inf
  else:
    R = np.identity(STATE_SIZE) # KF Process uncertainty/noise
    R[0,0] = r_xy # x
    R[1,1] = r_xy # y
    R[2,2] = 0.0000000001 # z
    R[3,3] = r_ori
    R[4,4] = r_wl # x_size
    R[5,5] = r_wl # y_size
    R[6,6] = 0.0000000001 
    R[7,7] = r_v # v_x
    R[8,8] = r_v # v_y
    R[9,9] = 0.0000000001 # v_z should be zero # TODO check that the order of Q is correct
    R[10,10] = r_a
    R[11,11] = r_a
    R[12,12] = 0.0000000001
    R[13,13] = r_ori_dot

    Q = np.zeros((STATE_SIZE, STATE_SIZE))
    Q[10,10] = q_a
    Q[11,11] = q_a
    Q[13,13] = r_ori_dot

    # print "get_MOT_score params:",params
    # print "get_MOT_score Q:",Q

    for label_i, labels_dir_path in enumerate(labels_paths_v):
      labels_json_path = labels_dir_path+'/'+listdir(labels_dir_path)[0] # FIXME no checks done if there isnt exactly only one label or if is a json
      pixor_json_name = high_set_v[label_i] + "/pixor_outputs.json"
      pixor_stats_json = pixor_json_name[0:len(pixor_json_name)-5]+"_stats.json"
      fused_pose_json = high_set_v[label_i] + "/fused_pose/fused_pose.json"

      total_list = get_tracker_json(pixor_json_name=pixor_json_name,pixor_stats_json=None, tracker_json_outfile=None, fused_pose_json=fused_pose_json, max_age=max_age,min_hits=min_hits,hung_thresh=ha, Q=Q, R=R, is_write=False)

      MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = \
      check_iou_json(labels_json_path, None, 100., "IOU", is_write=False, total_list=total_list)
      MOTA *= 100.
      if is_print:
        print MOTA, MOTP, MOTA-MOTP

      overall_MOTP += MOTP
      overall_MOTA += MOTA

  overall_MOTA /= len(labels_paths_v)
  overall_MOTP /= len(labels_paths_v)
  return overall_MOTA-overall_MOTP


'''
@brief: coordinate descent method in the paper "Discriminative Training of Kalman Filter"
@param num_params: number of parameters to search for
@param fn: function call
@param init_params: starting position for thoptional
@param max_iter
@param min_alpha
@outputs: true if it converges and false otherwise
'''
@jit
def coord_search(max_iter, min_alpha, high_set_v,labels_paths_v):
  # try for all params except for the ages because they are not coninuous. random init pts
  # params: R: x/y phi w/l v a phi_dot; Q: qv,qp; ha
  
  best_score = -np.inf
  best_params = []
  best_maxage = None
  best_minhits = None

  for max_age in range(1,10,1):
    for min_hits in range(1,10,1):
      num_params = 9
      alpha_ps = np.ones(num_params)*100.
      alpha_ps[-1] = 2.# ha
      init_params=[0.01,0.1,10.**-5,0.01,0.05,0.1,0.1,0.1,0.05]
      print "iteration:", max_age, min_hits
      is_conv, params =coord_descent(num_params=num_params, fn=get_MOT_score, ALPHA_PS=alpha_ps, dec_alpha=0.5, max_iter=10**3, 
                    min_alpha=0.5, init_params=init_params, fn_params=(high_set_v, labels_paths_v, max_age,min_hits))
      print "is converges:", is_conv
      print "best params of iteration:", params
      score = get_MOT_score(params, high_set_v, labels_paths_v, max_age,min_hits)
      print "best score:", score
      if score > best_score:
        best_score = score
        best_params = params
        best_maxage=max_age
        best_minhits=min_hits
  print "best:", best_score, best_params, best_maxage, best_minhits
  get_MOT_score(best_params, high_set_v, labels_paths_v, best_maxage,best_minhits,is_print=True)

  return is_conv

if __name__ == '__main__':
  # distance_metric = "IOU" # using IOU as distance metric
  # thres_d = 100. # threshold distance to count as a correspondance, beyond it will be considered as missed detection
  # TODO test with other distance metrics and thresholds

  labels_paths_v = []
  high_set_v = []

  # parent_dir = "/home/yl/Downloads/raw_data/"
  parent_dir = "/media/yl/demo_ssd/raw_data/"
  for root, dirs, files in walk(parent_dir):
    # identify where sets are using the "labels" directories
    for i, dire in enumerate(dirs):
      if dire == "labels":
        labels_dir=join(root,dire)
        low_set_dir= join(labels_dir, pardir) # location of the log_low set

        # look for the dir wt the name 
        try:
          set_contents = listdir(low_set_dir)
          if "pixor_train" in set_contents:
              print "found set in log low at "




        set_name =  root.split('/')[-1] # set name

        # loc of log_high set
        high_set_dir=join(low_set_dir, pardir) 
        high_set_dir=join(high_set_dir, pardir) 
        high_set_dir=join(high_set_dir, "log_high") 
        high_set_dir=join(high_set_dir, set_name) 

        '''
        # check if set dir has pcds and fused_poses
        try:
          set_contents = listdir(high_set_dir)
          if "fused_pose" in set_contents and "pcds" in set_contents:
            print "found set", high_set_dir
            high_set_v.append(high_set_dir)
            labels_paths_v.append(labels_dir)
        except:
          print "labels found but missing other contents at", high_set_dir
        '''

  # print high_set_v
  # print labels_paths_v
  
  coord_search(10.**3, 1.0, high_set_v,labels_paths_v)

  print "Done."

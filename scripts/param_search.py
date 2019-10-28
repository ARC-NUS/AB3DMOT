#!/usr/bin/env python

# file used to get best params for tracker
import numpy as np
from os import listdir
from os.path import isfile, join
import csv

from read_json_jic import get_tracker_json
from check_iou_jsons import check_iou_json
from coord_desc import coord_descent
from numba import prange, jit
import threading
from multiprocessing.pool import ThreadPool
import datetime
from yl_utils import STATE_SIZE

@jit(parallel=True)
def loop_ha(pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits, pixor_json_name,pixor_stats_json,fused_pose_json,labels_json_path, thres_d, distance_metric, best_list, best_i):
  q_arr = [-1.,0.,1.]
  q_v_arr = [-10,-7.,-5]
  thres_d = 100.
  distance_metric = "IOU"
  min_MOTP = float('inf')
  max_MOTA = float('inf') * -1.
  best_MOTA = None
  best_MOTP = None
  best_MOT = float('inf') * -1.
    
  q_xy = pq_xy
  q_wx = q_arr[pq_wx]
  q_ly = q_arr[pq_ly]
  q_v = q_v_arr[pq_v]
  q_heading = pq_heading-3
   
  max_age = pmax_age+2
  min_hits = pmin_hits+2
  
  ha_arr = [0.01, 0.025, 0.05] 
  
  for ha_iter in range(len(ha_arr)):
    ha_thresh = ha_arr[ha_iter]
    
    Q = np.identity(10) # KF Process uncertainty/noise
    Q[0,0] = 10.**q_xy # x
    Q[1,1] = 10.**q_xy # y
    Q[2,2] = 0.0000000001 # z
    Q[3,3] = 10.**q_heading
    Q[4,4] = 10.**q_wx # x_size
    Q[5,5] = 10.**q_ly # y_size
    Q[6,6] = 0.0000000001 
    Q[7,7] = 10.**q_v # v_x
    Q[8,8] = 10.**q_v # v_y
    Q[9,9] = 0.0000000001 # v_z should be zero # TODO check that the order of Q is correct
    
    total_list = get_tracker_json(pixor_json_name=pixor_json_name,pixor_stats_json=pixor_stats_json, tracker_json_outfile=None, fused_pose_json=fused_pose_json, max_age=max_age,min_hits=min_hits,hung_thresh=ha_thresh, Q=Q, is_write=False)
    
    MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = \
    check_iou_json(labels_json_path, None, thres_d, distance_metric, is_write=False, total_list=total_list)
    
    MOTA *= 100.
    best_MOT = MOTA-MOTP
            
    # send it back to the main thread
    best_list[best_i] = [best_MOT, MOTP, MOTA, pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits, ha_thresh]
            

@jit(parallel=True)
def grid_search(distance_metric, thres_d, labels_json_path, pixor_json_name, fused_pose_json, pixor_stats_json):
  pmin_hits_size = 4
  pmax_age_size = 4
  overall_mota = float('inf')
  overall_motp = float('inf') * -1.
  overall_MOT = float('inf') * -1.
  q_range = 3
  best_MOT_score = -float('inf')
  best_MOT_hp = [] # TODO remove size hardcode? [best_MOT, MOTP, MOTA, pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits, ha_thresh]

  for pq_xy in range(q_range):
    for pq_wx in range(q_range):
      pq_ly = pq_wx
#       for pq_ly in range(q_range):
      for pq_v in range(q_range):
        for pq_heading in range(q_range):
          for pmax_age in range(pmax_age_size):
            print [pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age]
            print(datetime.datetime.now())
            threads = []
            pmin_hits_list = [None] * pmin_hits_size
            for pmin_hits in range(pmin_hits_size):
#                   print "start pmin hits", pmin_hits
              x = threading.Thread(target=loop_ha, args=(pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits,pixor_json_name,pixor_stats_json,fused_pose_json,labels_json_path, thres_d, distance_metric, pmin_hits_list, pmin_hits)) # run clear-mot calcs
              threads.append(x)
              x.start()
            for x in threads:
              x.join()
            
            # compare the MOTA-MOTP for the permutations of pmin and ha
            for i, mot_list in enumerate(pmin_hits_list):
#                 print mot_list
              mot = mot_list[0]
              if mot > best_MOT_score:
                best_MOT_score = mot
                best_MOT_hp = [] # TODO change to a range?
              if mot == best_MOT_score:  # TODO change to a range?
                best_MOT_hp.append(pmin_hits_list[i])
                print "best mot" , best_MOT_hp
                    
  print "final best mot:" , best_MOT_hp
  
'''              
@jit(parallel=True)
def fine_grid_search(distance_metric, thres_d, labels_json_path, pixor_json_name, fused_pose_json):
  pmin_hits_size = 1
  pmax_age_size = 1
  overall_mota = float('inf')
  overall_motp = float('inf') * -1.
  overall_MOT = float('inf') * -1.
  q_range = 3
  pq_xy =1
  pq_v = 0
  pmax_age = 1
  pmin_hits = 1
  pq_heading = 2
           
  for pq_wx in range(q_range):
    for pq_ly in range(q_range):
      print [pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age]
      print(datetime.datetime.now())
      loop_ha(pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits,pixor_json_name,fused_pose_json,labels_json_path, thres_d, distance_metric)
'''


'''
@brief: parallel search with the correct Q
@param delta_t: time between predictions
@output: best mot score anbd its params
@qv: covariance of the velocity in the const vel motion model
'''
def parallel_qv(pixor_json_name,pixor_stats_json, fused_pose_json, labels_json_path,delta_t=0.05):

  for min_age in range(1,6,1):
    for max_age in range(1,6,1):
      for ha in range(0.1,0.8,0.1):
        for qv_i in range(8):
          q_v = 10.**(qv_i-4)
          Q = np.zeros((STATE_SIZE, STATE_SIZE))
          Q[0,0]=delta_t**3*q_v/3.
          Q[1,1]=delta_t**3*q_v/3.
          Q[0,7]=delta_t**2*q_v/2.
          Q[1,8]=delta_t**2*q_v/2.
          Q[7,0]=delta_t**2*q_v/2.
          Q[8,1]=delta_t**2*q_v/2.
          Q[7,7]=delta_t*q_v
          Q[8,8]=delta_t*q_v

          total_list = get_tracker_json(pixor_json_name=pixor_json_name,pixor_stats_json=pixor_stats_json, tracker_json_outfile=None, fused_pose_json=fused_pose_json, max_age=max_age,min_hits=min_hits,hung_thresh=ha, Q=Q, is_write=False)

          MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = \
          check_iou_json(labels_json_path, None, 100., "IOU", is_write=False, total_list=total_list)
          MOTA *= 100.
          if is_print:
            print MOTA, MOTP, MOTA-MOTP


# params: xy, wl, v, ori, ha
def get_MOT_score(params, pixor_json_name,pixor_stats_json, fused_pose_json, labels_json_path, max_age,min_hits, is_print = True):

  # print params, pixor_json_name,pixor_stats_json, fused_pose_json, labels_json_path, max_age,min_hits

  # check validity of param and give really bad score if not valid

  q_xy = params[0]
  q_wl = params[1]
  q_v = params[2]
  q_ori = params[3]
  ha = params[4]

  is_params_ok = True
  if q_xy <= 0.:
    is_params_ok = False
  if q_wl <= 0.:
    is_params_ok = False
  if q_v <= 0.:
    is_params_ok = False
  if q_ori <= 0.:
    is_params_ok = False
  if ha <= 0. or ha >= 1.:
    is_params_ok = False

  if not is_params_ok:
    return -np.inf
  else:
    
    Q = np.identity(10) # KF Process uncertainty/noise
    Q[0,0] = q_xy # x
    Q[1,1] = q_xy # y
    Q[2,2] = 0.0000000001 # z
    Q[3,3] = q_ori
    Q[4,4] = q_wl # x_size
    Q[5,5] = q_wl # y_size
    Q[6,6] = 0.0000000001 
    Q[7,7] = q_v # v_x
    Q[8,8] = q_v # v_y
    Q[9,9] = 0.0000000001 # v_z should be zero # TODO check that the order of Q is correct

    # print "get_MOT_score params:",params
    # print "get_MOT_score Q:",Q
    total_list = get_tracker_json(pixor_json_name=pixor_json_name,pixor_stats_json=pixor_stats_json, tracker_json_outfile=None, fused_pose_json=fused_pose_json, max_age=max_age,min_hits=min_hits,hung_thresh=ha, Q=Q, is_write=False)

    MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = \
    check_iou_json(labels_json_path, None, 100., "IOU", is_write=False, total_list=total_list)
    MOTA *= 100.
    if is_print:
      print MOTA, MOTP, MOTA-MOTP
  return MOTA-MOTP


'''
@brief: coordinate descent method in the paper "Discriminative Training of Kalman Filter"
@param num_params: number of parameters to search for
@param fn: function call
@param init_params: starting position for thoptional
@param max_iter
@param min_alpha
@outputs: true if it converges and false otherwise
'''
def coord_search(max_iter, min_alpha, pixor_json_name,pixor_stats_json, fused_pose_json, labels_json_path):
  # try for all params except for the ages because they are not coninuous. random init pts
  # params: xy, wl, v, ori, ha
  
  best_score = -np.inf
  best_params = []
  best_maxage = None
  best_minhits = None

  for max_age in range(1,6,1):
    for min_hits in range(1,6,1):
      num_params = 5
      alpha_ps = np.ones(num_params)*100.
      alpha_ps[4] = 2.# ha
      init_params=[0.01,0.1,10.**-5,0.01,0.05]
      print "iteration:", max_age, min_hits
      is_conv, params =coord_descent(num_params=num_params, fn=get_MOT_score, ALPHA_PS=alpha_ps, dec_alpha=0.5, max_iter=10**3, 
                    min_alpha=1., init_params=init_params, fn_params=(pixor_json_name,pixor_stats_json, fused_pose_json, labels_json_path, max_age,min_hits))
      print "is converges:", is_conv
      print "best params of iteration:", params
      score = get_MOT_score(params, pixor_json_name,pixor_stats_json, fused_pose_json, labels_json_path, max_age,min_hits)
      print "best score:", score
      if score > best_score:
        best_score = score
        best_params = params
        best_maxage=max_age
        best_minhits=min_hits
  print "best:", best_score, best_params, best_maxage, best_minhits
  get_MOT_score(best_params, pixor_json_name,pixor_stats_json, fused_pose_json, labels_json_path, best_maxage,best_minhits,is_print=True)

  return is_conv



if __name__ == '__main__':
  # distance_metric = "IOU" # using IOU as distance metric
  # thres_d = 100. # threshold distance to count as a correspondance, beyond it will be considered as missed detection
  # TODO test with other distance metrics and thresholds
  
  # jsons
  # 2 Hz labels
  labels_json_path = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_annotations.json"
  # 20 hz pixor outputs:
  pixor_json_name = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs_tf_epoch_3_valloss_0.0093_2.json"
  # generated pixor stats file
  pixor_stats_json = pixor_json_name[0:len(pixor_json_name)-5]+"_stats.json"
  # 20 Hz fuse pose
  fused_pose_json = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/fused_pose/fused_pose.json"
  
  # grid_search(distance_metric, thres_d, labels_json_path, pixor_json_name, fused_pose_json, pixor_stats_json)
  
  # coord_search(10.**3, 1.0, pixor_json_name,pixor_stats_json, fused_pose_json, labels_json_path)

  parallel_qv(pixor_json_name,pixor_stats_json, fused_pose_json, labels_json_path,delta_t=0.05):


  print "Done."
  
  
  
  
      
      

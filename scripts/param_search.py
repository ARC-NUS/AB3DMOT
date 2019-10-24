#!/usr/bin/env python

# file used to get best params for tracker
import numpy as np
from os import listdir
from os.path import isfile, join
import csv

from read_json_jic import get_tracker_json
from check_iou_jsons import check_iou_json
from numba import prange, jit
import threading
from multiprocessing.pool import ThreadPool
import datetime

@jit(parallel=True)
def loop_ha(pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits, pixor_json_name,fused_pose_json,labels_json_path, thres_d, distance_metric, best_list, best_i):
  q_arr = [-10.,0.,10.]
  thres_d = 100.
  distance_metric = "IOU"
  min_MOTP = float('inf')
  max_MOTA = float('inf') * -1.
  best_MOTA = None
  best_MOTP = None
  best_MOT = float('inf') * -1.
    
  q_xy = pq_xy*2+1
  q_wx = q_arr[pq_wx]
  q_ly = q_arr[pq_ly]
  q_v = -pq_v*2-1
  q_heading = -pq_heading
   
  max_age = pmax_age+5
  min_hits = pmin_hits+2
  
  ha_arr = [0.025] 
  
  for ha_iter in range(1): # FIXME check the best for the ha and only return the best
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
    
    total_list = get_tracker_json(pixor_json_name=pixor_json_name, tracker_json_outfile=None, fused_pose_json=fused_pose_json, max_age=max_age,min_hits=min_hits,hung_thresh=ha_thresh, Q=Q, is_write=False)
    
    
    MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = \
    check_iou_json(labels_json_path, None, thres_d, distance_metric, is_write=False, total_list=total_list)
    
    MOTA *= 100.
    if MOTP is not None:
      if min_MOTP > MOTP:
        min_MOTP = MOTP
    
    if MOTA is not None:
      if max_MOTA < MOTA:
        max_MOTA = MOTA

    best_MOT = MOTA-MOTP # FIXME: also keep track of the corresp MOTA n MOTP
#     if MOTA > 82. and MOTP < 86.5:
#         if MOTA-MOTP > best_MOT: 
#             print "best MOT: ", [best_MOT, MOTP, MOTA]
#             print [pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits, ha_thresh]
            
    # send it back to the main thread
    best_list[best_i] = [best_MOT, MOTP, MOTA, pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits, ha_thresh]
            

@jit(parallel=True)
def grid_search(distance_metric, thres_d, labels_json_path, pixor_json_name, fused_pose_json, pixor_stats_json):
  pmin_hits_size = 3
  pmax_age_size = 3
  overall_mota = float('inf')
  overall_motp = float('inf') * -1.
  overall_MOT = float('inf') * -1.
  q_range = 3
  best_MOT_score = -float('inf')
  best_MOT_hp = [] # TODO remove size hardcode? [best_MOT, MOTP, MOTA, pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits, ha_thresh]

  for pq_xy in range(q_range):
    for pq_wx in range(q_range):
      for pq_ly in range(q_range):
        for pq_v in range(q_range):
          for pq_heading in range(q_range):
            for pmax_age in range(pmax_age_size):
              print [pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age]
              print(datetime.datetime.now())
              threads = []
              pmin_hits_list = [None] * pmin_hits_size
              for pmin_hits in range(pmin_hits_size):
#                   print "start pmin hits", pmin_hits
                x = threading.Thread(target=loop_ha, args=(pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits,pixor_json_name,fused_pose_json,labels_json_path, thres_d, distance_metric, pmin_hits_list, pmin_hits)) # run clear-mot calcs
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
  
                 
@jit(parallel=True)
def fine_grid_search(distance_metric, thres_d, labels_json_path, pixor_json_name, fused_pose_json):
  pmin_hits_size = 3
  pmax_age_size = 3
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
         




def no_write(tracker_json_file, labels_json_path, pixor_json_name, fused_pose_json):
  
  # run clear-mot calcs
  distance_metric = "IOU" # using IOU as distance metric
  thres_d = 100. # threshold distance to count as a correspondance, beyond it will be considered as missed detection
  # TODO test with other distance metrics and thresholds
  
#   output_files = [f for f in listdir(tracker_json_dir) if isfile(join(tracker_json_dir, f))]
  output_files.sort()
  min_MOTP = float('inf')
  max_MOTA = float('inf') * -1.
  min_MOTP_file = None
  max_MOTA_file = None
  best_MOT = float('inf') * -1.
  best_MOT_file = None
  best_MOTA = None
  best_MOTP = None
  
  
  age = (output_file.split('age')[1]).split('_')[0]
  hits = (output_file.split('hits')[1]).split('_')[0]
  hung_arr = (output_file.split('thresh')[1]).split('.')[0:2]
  hung_thresh = hung_arr[0] + "." + hung_arr[1]
  
  # TODO fix when doing doubles
  q_xy = (output_file.split('_xy')[1]).split('.')[0]
  q_head = (output_file.split('_ori')[1]).split('.')[0]
  q_wx = (output_file.split('wx')[1]).split('.')[0]
  q_ly = (output_file.split('ly')[1]).split('.')[0]
  q_v = (output_file.split('_v')[1]).split('.')[0]
  
  MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = \
    check_iou_json(labels_json_path, tracker_json_dir+output_file, thres_d, distance_metric)
  csv_writer.writerow([output_file,age, hits, hung_thresh, q_xy, q_head, q_wx, q_ly, q_v, MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt, MOTA-MOTP])
  
  if MOTP is not None:
    if min_MOTP > MOTP:
      min_MOTP = MOTP
      min_MOTP_file = output_file
  
  if MOTA is not None:
    if max_MOTA < MOTA:
      max_MOTA = MOTA
      max_MOTA_file = output_file
      
  if MOTA > 0.5 and MOTP < 0.5:
      if MOTA-MOTP > best_MOT: 
          best_MOT = MOTA-MOTP
          best_MOT_file = output_file
  
#   print "done with ", output_file
  print "max_MOTA: ", max_MOTA, " from file: ", max_MOTA_file
  print" min MOTP", min_MOTP, "from file: ", min_MOTP_file
  if best_MOT_file is not None:
    print "best MOT: ", best_MOTA, best_MOTP, best_MOT_file
  
  return MOTA, MOTP
                    
  
def write_csv(tracker_json_dir, labels_json_path, pixor_json_name, fused_pose_json):
  # run clear-mot calcs
  distance_metric = "IOU" # using IOU as distance metric
  thres_d = 100. # threshold distance to count as a correspondance, beyond it will be considered as missed detection
  # TODO test with other distance metrics and thresholds
  
#   output_files = [f for f in listdir(tracker_json_dir) if isfile(join(tracker_json_dir, f))]
  output_files.sort()
  min_MOTP = float('inf')
  max_MOTA = float('inf') * -1.
  min_MOTP_file = None
  max_MOTA_file = None
  best_MOT = float('inf') * -1.
  best_MOT_file = None
  best_MOTA = None
  best_MOTP = None
  
  with open('mot_results_Q.csv', mode='w') as mot_file:
    csv_writer = csv.writer(mot_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["output_file","age", "hits", "hung_thresh", "q_xy", "q_head", "q_wx", "q_ly", "q_v", "MOTA", "MOTP", "total_dist", "total_ct", "total_mt", "total_fpt", "total_mmet", "total_gt", "MOTA-MOTP"])
    for output_file in output_files:
      age = (output_file.split('age')[1]).split('_')[0]
      hits = (output_file.split('hits')[1]).split('_')[0]
      hung_arr = (output_file.split('thresh')[1]).split('.')[0:2]
      hung_thresh = hung_arr[0] + "." + hung_arr[1]
      
      # TODO fix when doing doubles
      q_xy = (output_file.split('_xy')[1]).split('.')[0]
      q_head = (output_file.split('_ori')[1]).split('.')[0]
      q_wx = (output_file.split('wx')[1]).split('.')[0]
      q_ly = (output_file.split('ly')[1]).split('.')[0]
      q_v = (output_file.split('_v')[1]).split('.')[0]
      
      MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = \
        check_iou_json(labels_json_path, tracker_json_dir+output_file, thres_d, distance_metric)
      csv_writer.writerow([output_file,age, hits, hung_thresh, q_xy, q_head, q_wx, q_ly, q_v, MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt, MOTA-MOTP])
      
      if MOTP is not None:
        if min_MOTP > MOTP:
          min_MOTP = MOTP
          min_MOTP_file = output_file
      
      if MOTA is not None:
        if max_MOTA < MOTA:
          max_MOTA = MOTA
          max_MOTA_file = output_file
          
      if MOTA > 0.5 and MOTP < 0.5:
          if MOTA-MOTP > best_MOT: 
              best_MOT = MOTA-MOTP
              best_MOT_file = output_file
      
      print "done with ", output_file
    print "max_MOTA: ", max_MOTA, " from file: ", max_MOTA_file
    print" min MOTP", min_MOTP, "from file: ", min_MOTP_file
    print "best MOT: ", best_MOTA, best_MOTP, best_MOT_file

if __name__ == '__main__':
  
  # 20 hz pixor outputs:
#   pixor_json_name = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs.json"
#   pixor_json_name = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs_tf_epoch_3_valloss_0.0093_2.json"
#   fused_pose_json = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/fused_pose/fused_pose.json"
#   
#   # 20 hz tracker outputs:
#   tracker_json_dir = "/media/yl/downloads/tracker_results/set_7/tracker_results_age3_hits2_thresh_0.005/"
#   # 2 hz labels:
#   labels_json_path = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_annotations.json"
# 
#   is_gen_tracks = False # FIXME permutate Q values here
  # TODO get better param range
  # generate tracker files
#   if(is_gen_tracks):
#     p_grid_search()


  distance_metric = "IOU" # using IOU as distance metric
  thres_d = 100. # threshold distance to count as a correspondance, beyond it will be considered as missed detection
  # TODO test with other distance metrics and thresholds
  
  # jsons
  labels_json_path = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_annotations.json"
  pixor_json_name = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs_tf_epoch_3_valloss_0.0093_2.json"
  fused_pose_json = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/fused_pose/fused_pose.json"

  pixor_stats_json = pixor_json_name[0:len(pixor_json_name)-5]+"_stats.json"
  if no_write:
    grid_search(distance_metric, thres_d, labels_json_path, pixor_json_name, fused_pose_json, pixor_stats_json)
  print "Done."
  
  
  
  
      
      
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

@jit
def loop_ha(pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age_arr, pmax_age, pmin_hits_arr, pmin_hits):
  dummy_files = ["/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy0.json",
                  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy1.json",
                  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy2.json",
                  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy3.json",
                  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy4.json",
                  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy5.json",
                  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy6.json"]
  thres_d = 100.
  distance_metric = "IOU"
  min_MOTP = float('inf')
  max_MOTA = float('inf') * -1.
  best_MOTA = None
  best_MOTP = None
  best_MOT = float('inf') * -1.
    
  q_xy = pq_xy-1
  q_wx = pq_wx-1 
  q_ly = pq_ly-1
  q_v = pq_v-1
  q_heading = pq_heading-1
   
  max_age = pmax_age+1
  min_hits = pmin_hits+1
  for ha_thresh in np.arange(0.1,1.0,0.4):
    # TODO param search wt HA thresh?
#                 ha_thresh = ha_thresh_exp
#                     tracker_params = "age" + str(max_age) + "_hits" + str(min_hits) +"_thresh" + str(ha_thresh)
    
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
    
#                     q_params = "_xy" + str(q_xy) + "_ori" + str(q_heading) + "_wx" + str(q_wx) + "_ly" + str(q_ly) + "_v" +  str(q_v)
#                     tracker_json_outfile = "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/tracker_px_stats_" + tracker_params +"_Q"+ q_params + ".json"
    dummy_file =  dummy_files[pmin_hits]
#                     print tracker_params, q_params, tracker_json_outfile
    get_tracker_json(pixor_json_name=pixor_json_name, tracker_json_outfile=dummy_file, fused_pose_json=fused_pose_json, max_age=max_age,min_hits=min_hits,hung_thresh=ha_thresh, Q=Q)
    
    
    MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = \
    check_iou_json(labels_json_path, dummy_file, thres_d, distance_metric)
    
    
    if MOTP is not None:
      if min_MOTP > MOTP:
        min_MOTP = MOTP
#                         min_MOTP_file = tracker_json_outfile
#         print" min MOTP", min_MOTP
#         print pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits, ha_thresh
    
    if MOTA is not None:
      if max_MOTA < MOTA:
        max_MOTA = MOTA
#                         max_MOTA_file = tracker_json_outfile
#         print "max_MOTA: ", max_MOTA
#         print pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits, ha_thresh
        
    if MOTA > 0.6 and MOTP < 80.:
        if MOTA-MOTP > best_MOT: 
            best_MOT = MOTA-MOTP # FIXME: also keep track of the corresp MOTA n MOTP
#                             best_MOT_file = tracker_json_outfile
            print "best MOT: ", best_MOTA, best_MOTP, tracker_json_outfile
            print pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age, pmin_hits, ha_thresh

  pmin_hits_arr[pmin_hits,0] = max_MOTA
  pmin_hits_arr[pmin_hits,1] = min_MOTP
  pmin_hits_arr[pmin_hits,2] = best_MOT
#   print "pmin hits", pmin_hits, pmin_hits_arr
#   print "end pmin hits", pmin_hits

@jit(parallel=True)
def p_grid_search():
  
  # TODO move this to main
  # run clear-mot calcs
  distance_metric = "IOU" # using IOU as distance metric
  thres_d = 100. # threshold distance to count as a correspondance, beyond it will be considered as missed detection
  # TODO test with other distance metrics and thresholds
  labels_json_path = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_annotations.json"
  pixor_json_name = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs.json"
  fused_pose_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/fused_pose/fused_pose.json"

#   min_MOTP_file = None
#   max_MOTA_file = None
#   best_MOT_file = None
  
  pmin_hits_size = 5
  pmax_age_size = 5
  overall_mota = float('inf')
  overall_motp = float('inf') * -1.
  overall_MOT = float('inf') * -1.
  q_range = 5
  dummy_files = ["/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy0.json",
                  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy1.json",
                  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy2.json",
                  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy3.json",
                  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy4.json",
                  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy5.json",
                  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy6.json"]
  
  for pq_xy in range(q_range):
      for pq_wx in range(q_range):
        for pq_ly in range(q_range):
          for pq_v in range(q_range):
            for pq_heading in range(q_range):
              pmax_age_arr = np.zeros((pmax_age_size, 3)) # mota, motp, mot
              for pmax_age in range(pmax_age_size):
                pmin_hits_arr = np.zeros((pmin_hits_size, 3)) # mota, motp, mot
                threads = []
                for pmin_hits in range(pmin_hits_size):
#                   print "start pmin hits", pmin_hits
                  x = threading.Thread(target=loop_ha, args=(pq_xy, pq_wx, pq_ly, pq_v, pq_heading, pmax_age_arr, pmax_age, pmin_hits_arr, pmin_hits))
                  threads.append(x)
                  x.start()
                for x in threads:
                    x.join()
#                 pmax_age_arr[pmax_age,0] = max(pmin_hits_arr[:,0])
#                 pmax_age_arr[pmax_age,1] = min(pmin_hits_arr[:,1])
#                 pmax_age_arr[pmax_age,2] = max(pmin_hits_arr[:,2])
#                 
#               if overall_mota < max(pmax_age_arr[:,0]):
#                 overall_mota = max(pmax_age_arr[:,0])
#                 print "max_MOTA: ", overall_mota
#               if overall_motp > min(pmax_age_arr[:,1]):
#                 overall_motp = min(pmax_age_arr[:,1])
#                 print "min_MOTP: ", overall_motp
#               if overall_MOT < max(pmax_age_arr[:,2]):
#                 overall_MOT = max(pmax_age_arr[:,2])
#                 print "best MOT: ", overall_MOT
#                 print "done with ", tracker_json_outfile
#   print "max_MOTA: ", max_MOTA, " from file: ", max_MOTA_file
#   print" min MOTP", min_MOTP, "from file: ", min_MOTP_file
#   print "best MOT: ", best_MOTA, best_MOTP, best_MOT_file
#   print "max_MOTA: ", max_MOTA
#   print" min MOTP", min_MOTP
#   print "best MOT: ", best_MOT


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
  pixor_json_name = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs.json"
  fused_pose_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/fused_pose/fused_pose.json"
  
  # 20 hz tracker outputs:
  tracker_json_dir = "/media/yl/downloads/tracker_results/set_7/tracker_results_age3_hits2_thresh_0.005/"
  # 2 hz labels:
  labels_json_path = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_annotations.json"

  is_gen_tracks = False # FIXME permutate Q values here
  # TODO get better param range
  # generate tracker files
#   if(is_gen_tracks):
#     p_grid_search()
    
  if no_write:
    p_grid_search()
  print "Done."
  
  
  
  
      
      
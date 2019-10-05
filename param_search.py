#!/usr/bin/env python

# file used to get best params for tracker
import numpy as np
from os import listdir
from os.path import isfile, join
import csv

from read_json_jic import get_tracker_json
from check_iou_jsons import check_iou_json
from numba import prange, jit



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
  
  
  min_MOTP = float('inf')
  max_MOTA = float('inf') * -1.
  min_MOTP_file = None
  max_MOTA_file = None
  best_MOT = float('inf') * -1.
  best_MOT_file = None
  best_MOTA = None
  best_MOTP = None
  
  for pq_xy in prange(11):
      for pq_wx in prange(11):
        for pq_ly in prange(11):
          for pq_v in prange(11):
            for pq_heading in prange(11):
              for pmax_age in prange(4):
                for pmin_hits in prange(4):
                  for ha_thresh in np.arange(0.1,1.0,0.4):
                    
                    q_xy = pq_xy-5
                    q_wx = pq_wx-5 
                    q_ly = pq_ly-5
                    q_v = pq_v-5
                    q_heading = pq_heading-5
                    
                    max_age = pmax_age+2
                    min_hits = pmin_hits+2
                    
                    
                    # TODO param search wt HA thresh?
    #                 ha_thresh = ha_thresh_exp
                    tracker_params = "age" + str(max_age) + "_hits" + str(min_hits) +"_thresh" + str(ha_thresh)
                    
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
                    
                    q_params = "_xy" + str(q_xy) + "_ori" + str(q_heading) + "_wx" + str(q_wx) + "_ly" + str(q_ly) + "_v" +  str(q_v)
                    
                    tracker_json_outfile = "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/tracker_px_stats_" + tracker_params +"_Q"+ q_params + ".json"
                    dummy_file =  "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/dummy.json"
                    print tracker_params, q_params, tracker_json_outfile
                    get_tracker_json(pixor_json_name=pixor_json_name, tracker_json_outfile=dummy_file, fused_pose_json=fused_pose_json, max_age=max_age,min_hits=min_hits,hung_thresh=ha_thresh, Q=Q)
                    
                    
                    MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = \
                    check_iou_json(labels_json_path, dummy_file, thres_d, distance_metric)
                    
                    
                    if MOTP is not None:
                      if min_MOTP > MOTP:
                        min_MOTP = MOTP
                        min_MOTP_file = tracker_json_outfile
                        print" min MOTP", min_MOTP, "from file: ", min_MOTP_file
                    
                    if MOTA is not None:
                      if max_MOTA < MOTA:
                        max_MOTA = MOTA
                        max_MOTA_file = tracker_json_outfile
                        print "max_MOTA: ", max_MOTA, " from file: ", max_MOTA_file
                        
                    if MOTA > 0.5 and MOTP < 0.5:
                        if MOTA-MOTP > best_MOT: 
                            best_MOT = MOTA-MOTP
                            best_MOT_file = tracker_json_outfile
                            print "best MOT: ", best_MOTA, best_MOTP, best_MOT_file
                    
                    print "done with ", tracker_json_outfile
  print "max_MOTA: ", max_MOTA, " from file: ", max_MOTA_file
  print" min MOTP", min_MOTP, "from file: ", min_MOTP_file
  print "best MOT: ", best_MOTA, best_MOTP, best_MOT_file
                    

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
  if(is_gen_tracks):
    p_grid_search()
    
  if no_write:
    p_grid_search()
  
  
  
  
      
      
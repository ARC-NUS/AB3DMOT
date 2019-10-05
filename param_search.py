#!/usr/bin/env python

# file used to get best params for tracker
import numpy as np
from os import listdir
from os.path import isfile, join
import csv

from read_json_jic import get_tracker_json
from check_iou_jsons import check_iou_json


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
    for max_age in range (1,10):
      for min_hits in range(1,10):
        for hung_thresh in np.arange(0.01, 0.1, 0.01):
          tracker_params = "age" + str(max_age) + "_hits" + str(min_hits) + "_thresh" + str(hung_thresh)
          tracker_json_outfile = tracker_json_dir + "/tracker_" + tracker_params + ".json"
  
          # TODO param search the sensor model covariance as well
          
          get_tracker_json(pixor_json_name=pixor_json_name, tracker_json_outfile=tracker_json_outfile, fused_pose_json=fused_pose_json, \
                           max_age=max_age,min_hits=min_hits,hung_thresh=hung_thresh)

  # run clear-mot calcs
  distance_metric = "IOU" # using IOU as distance metric
  thres_d = 100. # threshold distance to count as a correspondance, beyond it will be considered as missed detection
  # TODO test with other distance metrics and thresholds
  
  output_files = [f for f in listdir(tracker_json_dir) if isfile(join(tracker_json_dir, f))]
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
      
      
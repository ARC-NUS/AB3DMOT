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
  pixor_json_name = "/media/yl/demo_ssd/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/16_sep/log_high/set_2/pixor_outputs.json"
  fused_pose_json = "/media/yl/demo_ssd/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/16_sep/log_high/set_2/fused_pose/fused_pose.json"
  
  # 20 hz tracker outputs:
  tracker_json_dir = "./json_results/2019-08-27-21-55-47_set2/"
  # 2 hz labels:
  labels_json_path = "/media/yl/demo_ssd/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/16_sep/log_low/set_2/set2_annotations.json"

  is_gen_tracks = True
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
  
  with open('mot_results.csv', mode='w') as mot_file:
    for output_file in output_files:
      age = (output_file.split('age')[1]).split('_')[0]
      hits = (output_file.split('hits')[1]).split('_')[0]
      hung_arr = (output_file.split('thresh')[1]).split('.')[0:2]
      hung_thresh = hung_arr[0] + "." + hung_arr[1]
      
      csv_writer = csv.writer(mot_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = \
        check_iou_json(labels_json_path, tracker_json_dir+output_file, thres_d, distance_metric)
      csv_writer.writerow([output_file,age, hits, hung_thresh, MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt])
      
      if MOTP is not None:
        if min_MOTP > MOTP:
          min_MOTP = MOTP
          min_MOTP_file = output_file
      
      if MOTA is not None:
        if max_MOTA < MOTA:
          max_MOTA = MOTA
          max_MOTA_file = output_file
      
      print "done with ", output_file
    print "max_MOTA: ", max_MOTA, " from file: ", max_MOTA_file
    print" min MOTP", min_MOTP, "from file: ", min_MOTP_file
      
      
#/usr/bin/env python

import argparse
from main import AB3DMOT
import json
import numpy as np

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='a baseline for 3D MOT.')
  parser.add_argument("input_set", help="input set to be used. must be a valid dir in ../data")
#   parser.add_argument("-j", "--json", action="store_true",
#                     help="read from json file (instead of kitti format)")

  args = parser.parse_args()
  result_sha = args.input_set
  
  det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist', 4:'Truck'}
  
  # default
#   tracker_params = "age3_hits2_thresh_05"
#   mot_tracker = AB3DMOT(is_jic=True,max_age=3,min_hits=2,hung_thresh=0.05)

  tracker_params = "age5_hits1_thresh_025"
  mot_tracker = AB3DMOT(is_jic=True,max_age=5,min_hits=1,hung_thresh=0.025)
  
#   tracker_params = "age3_hits2_thresh_075"
#   mot_tracker = AB3DMOT(is_jic=True,max_age=3,min_hits=2,hung_thresh=0.075)

  # load detetions
  json_name = "/media/yl/demo_ssd/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/16_sep/log_high/set_2/pixor_outputs.json"
  
  json_outfile = "/media/yl/demo_ssd/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/16_sep/log_high/set_2/tracker_" + tracker_params + ".json"
  # for a single frame
  # dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
  # we set zero for z & h for BEV tracking
  
  total_list=[]
  
  with open(json_name) as json_file:
    data = json.load(json_file, encoding="utf-8")
    for pcd in data:
      print("working on pcd: " + pcd["name"])
      dets = []
      add_info = []
      # TODO extract objects in frame
      for obj in pcd["objects"]:
        l = obj["width"] # louis did some weird shit
        w = obj["length"]
        h = 1.0
        x = obj["centroid"][0]
        y = obj["centroid"][1]
        z = 0.0
        theta = obj["heading"]
        dets.append([x,y,z,theta,l,w,h])
        add_info.append([])
        # TODO get the additional info for each frame
        # TODO use the probability to do updates
      
      np_dets = np.array(dets)
      if len(dets) == 0:
        np_dets = np_dets.reshape((len(dets),7))
        
      np_info = np.array(add_info)
      dets_all = {'dets': np_dets, 'info': np_info}
      # perform update
      trackers = mot_tracker.update(dets_all)
      
      result_trks=[]
      
      for d in trackers:
#         obj_dict={"width":d[5], "height": d[6], "length": d[4], "x": d[0], "y": d[1], "z": d[2], "yaw": d[3], "id": d[7]}
        obj_dict={"width":d[1], "height": d[0], "length": d[2], "x": d[3], "y": d[4], "z": d[5], "yaw": d[6], "id": d[7]}
        result_trks.append(obj_dict)
      total_list.append({"name": pcd["name"], "objects":result_trks})
  
  # parse into json
  with open(json_outfile, "w+") as outfile:
      json.dump(total_list, outfile, indent=1)

      
  print "Done"
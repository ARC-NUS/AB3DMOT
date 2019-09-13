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
  
  mot_tracker = AB3DMOT()
  
  # load detetions
  json_name = "/home/yl/bus_ws/src/AB3DMOT/data/KITTI/linn_jicetran_2019-08-27-22-47-10_set1/pixor_outputs.json"
  # for a single frame
#   dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
  # we set zero for z & h for BEV tracking
  
  with open(json_name) as json_file:
    data = json.load(json_file, encoding="utf-8")
    for pcd in data:
      print("working on pcd: " + pcd["name"])
      dets = []
      add_info = []
      # TODO extract objects in frame
      for obj in pcd["objects"]:
        w = obj["width"]
        l = obj["length"]
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
      
      for d in trackers:
        print d
      
      
  print "Done"
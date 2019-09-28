#/usr/bin/env python

import argparse
from main import AB3DMOT
import json
import numpy as np
from pyquaternion import Quaternion

def get_tracker_json(pixor_json_name, tracker_json_outfile, fused_pose_json, max_age=3,min_hits=2,hung_thresh=0.05, Q = np.identity(10)):
# we set zero for z & h for BEV tracking

  # x y z theta l w h 
  R = np.identity(7) # KF measurement uncertainty/noise
  
#   if (Q == None):
#       Q = np.identity(10) # KF Process uncertainty/noise
#       Q[7:,7:] *= 0.01
  
  # pixor stats LGCR set 7
  # x y w b theta
  # 194 22 358 [ 0.33482453  0.03826206 28.93938572 35.41910205  3.83654019]
  # tp at iou  0.75 :  0.351449275362
  R[0,0] = 0.33482453 
  R[1,1] = 0.03826206
  R[2,2] = 0.0001
  R[3,3] = 3.83654019
  R[4,4] = 28.93938572
  R[5,5] = 35.41910205
  R[6,6] = 0.0001
  
  mot_tracker = AB3DMOT(is_jic=True,max_age=max_age,min_hits=min_hits,hung_thresh=hung_thresh, R=R, Q=Q)
  
  det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist', 4:'Truck'}
  
  total_list=[]
  
  is_utm = True # to track in UTM instead of in baselink frame
  
  with open(pixor_json_name) as json_file:
    with open(fused_pose_json) as fp_json:
      data = json.load(json_file, encoding="utf-8")
      fp_data = json.load(fp_json, encoding="utf-8")
      fp_msgs = fp_data['ego_loc']
      

      for count, pcd in enumerate(data):
        fp_msg = fp_msgs[count]
        if fp_msg['counter'] != int(pcd["name"].split('.')[0]):
          print "error fused pose json and pixor json mismatched"
          print "offending fused pose message: ", fp_msg['counter']
          print "but expected: ", int(pcd["name"].split('.')[0])
          raise ValueError
        else:
          w_yaw_bl = float(fp_msg['pose']['attitude']['yaw'])
          w_q_bl = Quaternion(axis=[0, 0, 1], angle=w_yaw_bl)
        
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
          
          
          # transform x & y to UTM coordinates
          ## get corresponding fused_pose msg
          
          utm_x = w_q_bl.rotation_matrix[0,0] * x + \
                  w_q_bl.rotation_matrix[0,1] * y + \
                  fp_msg['pose']['position']['x']
          
          utm_y = w_q_bl.rotation_matrix[1,0] * x + \
                  w_q_bl.rotation_matrix[1,1] * y + \
                  fp_msg['pose']['position']['y']
#           print x,y, utm_x, utm_y
           # TODO check if utm transform is correct
          if is_utm:
            dets.append([utm_x,utm_y,z,theta,l,w,h])
          else:
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
          if is_utm:
            utm_x = d[3]
            utm_y = d[4]
            
            bl_x = w_q_bl.rotation_matrix[0,0] * (utm_x - fp_msg['pose']['position']['x']) + \
                   w_q_bl.rotation_matrix[1,0] * (utm_y - fp_msg['pose']['position']['y'])
                   
                   
            bl_y = w_q_bl.rotation_matrix[0,1] * (utm_x - fp_msg['pose']['position']['x']) + \
                   w_q_bl.rotation_matrix[1,1] * (utm_y - fp_msg['pose']['position']['y'])
                    
            #           print bl_x,bl_y, utm_x, utm_y
                            
            obj_dict={"width":d[1], "height": d[0], "length": d[2], "x": bl_x, "y": bl_y, "z": d[5], "yaw": d[6], "id": d[7]}
          else:
            obj_dict={"width":d[1], "height": d[0], "length": d[2], "x": d[3], "y": d[4], "z": d[5], "yaw": d[6], "id": d[7]}
          result_trks.append(obj_dict)
        total_list.append({"name": pcd["name"], "objects":result_trks})
  
  # parse into json
  with open(tracker_json_outfile, "w+") as outfile:
      json.dump(total_list, outfile, indent=1)

      
  print "Done"
  
  
  
if __name__ == '__main__':  
#   parser = argparse.ArgumentParser(description='a baseline for 3D MOT.')
#   parser.add_argument("input_set", help="input set to be used. must be a valid dir in ../data")
# #   parser.add_argument("-j", "--json", action="store_true",
# #                     help="read from json file (instead of kitti format)")

#   args = parser.parse_args()
#   result_sha = args.input_set
  
  # load detetions
  pixor_json_name = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs.json"
#   pixor_json_name = "/media/yl/demo_ssd/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/16_sep/log_high/set_2/pixor_outputs.json"

  # output json
  # default
  tracker_params = "age3_hits2_thresh_0.05"
  
  fused_pose_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/fused_pose/fused_pose.json"
  
  for i in range(0,10):
      Q = np.identity(10) # KF Process uncertainty/noise
      Q[0,0] = 0.01 # x
      Q[1,1] = 0.01 # y
      Q[2,2] = 0.0000000001 # z
      Q[3,3] = 0.01 # x_width
      Q[4,4] = 0.01 # y_width
      Q[5,5] = 0.0000000001 
      Q[6,6] = 0.01 # v_x
      Q[7,7] = 0.01 # v_y
#       Q[8,8] = 
      
      
      
      Q[7:,7:] *= 0.01 * 0.01 * (i+1) # (0.5**2)
      
      Q[:7,:7] *= 0.01 * 3 # 0.5 ** (5)
      
      tracker_json_outfile = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/tracker_px_stats_" + tracker_params + str(10*i)+ ".json"
      get_tracker_json(pixor_json_name=pixor_json_name, tracker_json_outfile=tracker_json_outfile, fused_pose_json=fused_pose_json, max_age=3,min_hits=2,hung_thresh=0.005, Q=Q)
      
      
#/usr/bin/env python

import argparse
from main import AB3DMOT
import json
import numpy as np
from pyquaternion import Quaternion
from numba import prange, jit


'''
@brief: Gets the MOT 
@param model: NN model to evaluate
@param pc_folder: Folder of the validation set
@return total_list: the list of all the object trackes for the whole input pixor json
'''
def get_tracker_json(pixor_json_name, tracker_json_outfile, fused_pose_json, max_age=3,min_hits=2,hung_thresh=0.05, Q = np.identity(10), is_write=True):
# we set zero for z & h for BEV tracking

  # x y z theta l w h 
  R = np.identity(7) # KF measurement uncertainty/noise
  
  # pixor stats LGCR set 7
  # tp, fp, fn, [x y w b theta]
  # pixor_outputs_tf_epoch_3_valloss_0.0093.json
  # 210 92 335 [0.03953874 0.00588307 0.02431999 0.39831919 0.00211127] precision@80%iou75 :  69.54%, recall:  54.69%
  R[0,0] = 0.03953874 #x
  R[1,1] = 0.00588307 #y
  R[2,2] = 0.0001 #z
  R[3,3] = 0.00211127 #theta
  R[4,4] = 0.39831919 #l
  R[5,5] = 0.02431999 #w
  R[6,6] = 0.0001 #h
  
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
        
#         print("working on pcd: " + pcd["name"])
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
          
        if is_check_online:
          # check MOTA and MOTP
          
          # TODO early stop?
          if MOT_score < acceptable:
            return None
            
            
        total_list.append({"name": pcd["name"], "objects":result_trks})
  
  # parse into json
  if is_write:
    with open(tracker_json_outfile, "w+") as outfile:
      json.dump(total_list, outfile, indent=1)


  return total_list
  
  
  
  
@jit(parallel=True)
def p_grid_search():
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
                    print tracker_params, q_params, tracker_json_outfile
                    get_tracker_json(pixor_json_name=pixor_json_name, tracker_json_outfile=tracker_json_outfile, fused_pose_json=fused_pose_json, max_age=max_age,min_hits=min_hits,hung_thresh=ha_thresh, Q=Q)
  return True

if __name__ == '__main__':  
#   parser = argparse.ArgumentParser(description='a baseline for 3D MOT.')
#   parser.add_argument("input_set", help="input set to be used. must be a valid dir in ../data")
# #   parser.add_argument("-j", "--json", action="store_true",
# #                     help="read from json file (instead of kitti format)")

#   args = parser.parse_args()
#   result_sha = args.input_set
  
  # load detetions
#   pixor_json_name = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs_tf_epoch_49_valloss_0.0107.json"
  pixor_json_name = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs_tf_epoch_3_valloss_0.0093_2.json"

#   pixor_json_name = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs.json"
  # output json
  # default
  
  
  fused_pose_json = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/fused_pose/fused_pose.json"
  
  is_tuning = False
  
  if is_tuning:
    p_grid_search()
  else:
    
    
    Q = np.identity(10) # KF Process uncertainty/noise
    
    q_xy = 0
    q_heading = -1.
    q_wx = -5.
    q_ly = -5.
    q_v = -1.
    max_age = 4
    min_hits = 4
    hung_thresh = 0.05
    tracker_params = "max_age="+str(max_age)+",min_hits="+str(min_hits)+",hung_thresh="+str(hung_thresh)
    
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
    
#     tracker_json_outfile = "/media/yl/downloads/tracker_results/set_7/tracker_results_age3_hits2_thresh_0.01/tracker_tf_epoch_36_valloss_0.0037_" + tracker_params +"_Q"+ q_params + ".json"
    tracker_json_outfile = "/media/yl/downloads/tracker_results/set_7/pixor_outputs_tf_epoch_3_valloss_0.0093_2" + tracker_params +"_Q"+ q_params + ".json"
    get_tracker_json(pixor_json_name=pixor_json_name, tracker_json_outfile=tracker_json_outfile, fused_pose_json=fused_pose_json, max_age=max_age,min_hits=min_hits,hung_thresh=hung_thresh, Q=Q)
            
      
      
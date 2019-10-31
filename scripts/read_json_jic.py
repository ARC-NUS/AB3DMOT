#/usr/bin/env python

import argparse
from main import AB3DMOT
import json
import numpy as np
from pyquaternion import Quaternion
from numba import prange, jit
from __builtin__ import False

import yl_utils 

'''
@brief: Gets the MOT 
@param model: NN model to evaluate
@param pc_folder: Folder of the validation set
@return total_list: the list of all the object trackes for the whole input pixor json
'''
def get_tracker_json(pixor_json_name, pixor_stats_json, tracker_json_outfile, fused_pose_json, max_age=3,min_hits=2,hung_thresh=0.05, Q = np.identity(10), is_write=True):
  is_check_online = False # FIXME
# we set zero for z & h for BEV tracking

  # print pixor_json_name, pixor_stats_json, tracker_json_outfile, fused_pose_json, max_age,min_hits,hung_thresh, Q, is_write
  # print "get_tracker_json Q:", Q
  R = yl_utils.px_stats_get_R(pixor_stats_json)
  P_0 = yl_utils.px_stats_get_P_0(pixor_stats_json)

  mot_tracker = AB3DMOT(is_jic=True,max_age=max_age,min_hits=min_hits,hung_thresh=hung_thresh, R=R, Q=Q, P_0 = P_0)
  
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
  pixor_json_name = "/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs_tf_epoch_3_valloss_0.0093_2.json"
  pixor_stats_json =  pixor_json_name[0:len(pixor_json_name)-5]+"_stats.json"
  fused_pose_json = "/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/fused_pose/fused_pose.json"
  
  is_tuning = False
  
  if is_tuning:
    p_grid_search()
  else:
    # Q = np.identity(yl_utils.STATE_SIZE) # KF Process uncertainty/noise
    # q_xy = 0
    # q_heading = -1.
    # q_wx = -5.
    # q_ly = -5.
    # q_v = -1.
    max_age = 4
    min_hits = 4
    hung_thresh = 0.05
    tracker_params = "max_age="+str(max_age)+",min_hits="+str(min_hits)+",hung_thresh="+str(hung_thresh)
    
    # Q[0,0] = 10.**q_xy # x
    # Q[1,1] = 10.**q_xy # y
    # Q[2,2] = 0.0000000001 # z
    # Q[3,3] = 10.**q_heading
    # Q[4,4] = 10.**q_wx # x_size
    # Q[5,5] = 10.**q_ly # y_size
    # Q[6,6] = 0.0000000001 
    # Q[7,7] = 10.**q_v # v_x
    # Q[8,8] = 10.**q_v # v_y
    # Q[9,9] = 0.0000000001 # v_z should be zero # TODO check that the order of Q is correct
    
    # q_params = "_xy" + str(q_xy) + "_ori" + str(q_heading) + "_wx" + str(q_wx) + "_ly" + str(q_ly) + "_v" +  str(q_v)

    qv=10.**-3
    q_params="qv_"+str(qv)
    if yl_utils.MOTION_MODEL == "CV":
      Q=yl_utils.get_CV_Q(qv,0.05)
    else:
      if yl_utils.MOTION_MODEL == "CA":
        Q=yl_utils.get_CA_Q(qv,0.05)
      else:
        print ("unknown motion model")
        raise ValueError
    
    tracker_json_outfile = "/home/yl/Downloads/tracker_results/set_7/ca_state_10" + tracker_params +"_Q"+ q_params + ".json"
    get_tracker_json(pixor_json_name=pixor_json_name, pixor_stats_json=pixor_stats_json, tracker_json_outfile=tracker_json_outfile, 
      fused_pose_json=fused_pose_json, max_age=max_age,min_hits=min_hits,hung_thresh=hung_thresh, Q=Q)
    print "Done"
      
      

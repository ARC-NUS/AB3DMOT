#/usr/bin/env python

import argparse
from main import AB3DMOT
import json
import numpy as np
from pyquaternion import Quaternion
from numba import prange, jit

import yl_utils 
import time

'''
@brief: Gets the MOT 
@param model: NN model to evaluate
@param pc_folder: Folder of the validation set
@return total_list: the list of all the object trackes for the whole input pixor json
'''
def get_tracker_json(pixor_json_name, tracker_json_outfile, fused_pose_json, pixor_stats_json=None, max_age=3,min_hits=2,hung_thresh=0.05, Q = np.identity(yl_utils.STATE_SIZE), R=np.identity(yl_utils.STATE_SIZE), is_write=True):
  is_check_online = False # FIXME
  
  p_v = 1000.
# we set zero for z & h for BEV tracking

  # print pixor_json_name, pixor_stats_json, tracker_json_outfile, fused_pose_json, max_age,min_hits,hung_thresh, Q, is_write
  # print "get_tracker_json Q:", Q
  if pixor_stats_json is not None:
    R = yl_utils.px_stats_get_R(pixor_stats_json)
    P_0 = yl_utils.px_stats_get_P_0(pixor_stats_json)
  else:
#     print "creating P_0 from R wt pv ", p_v
    P_0 = np.identity(yl_utils.STATE_SIZE) # KF measurement uncertainty/noise
    P_0[0:7,0:7] = R
    P_0[7,7] = p_v
    P_0[8,8] = p_v
    P_0[9,9] = p_v
    P_0[10,10] = p_v
    P_0[11,11] = p_v
    P_0[12,12] = p_v
    P_0[13,13] = p_v

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

          # FIXME what about for the theta???
          theta += w_yaw_bl
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
            
            bl_theta = d[6] - w_yaw_bl        
            #           print bl_x,bl_y, utm_x, utm_y
            

            obj_dict={"width":d[1], "height": d[0], "length": d[2], "x": bl_x, "y": bl_y, "z": d[5], "yaw": d[6], "id": d[7]}
          else:
            obj_dict={"width":d[1], "height": d[0], "length": d[2], "x": d[3], "y": d[4], "z": d[5], "yaw": bl_ori, "id": d[7]}
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
  
  

if __name__ == '__main__':  
  pixor_json_name = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_8/pixor_outputs_tf_epoch_3_valloss_0.0093.json"
  pixor_stats_json =  pixor_json_name[0:len(pixor_json_name)-5]+"_stats.json"
  fused_pose_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_8/fused_pose/fused_pose_new.json"
  
  '''
  pixor_json_name = "/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs_tf_epoch_3_valloss_0.0093_2.json"
  pixor_stats_json =  pixor_json_name[0:len(pixor_json_name)-5]+"_stats.json"
  fused_pose_json = "/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/fused_pose/fused_pose_new.json"
  '''

  is_tuning = False
  
  if is_tuning:
    p_grid_search()
  else:
    Q = np.identity(yl_utils.STATE_SIZE) # KF Process uncertainty/noise
    # q_xy = 0
    # q_heading = -1.
    # q_wx = -5.
    # q_ly = -5.
    # q_v = -1.
    max_age = 8
    min_hits = 6
    hung_thresh = 0.25
    tracker_params = "max_age="+str(max_age)+",min_hits="+str(min_hits)+",hung_thresh="+str(hung_thresh)
    
    '''
    params= [6.10351562e-01, 7.81250000e-02, 1.95312500e-08, 2.29588740e+11, 2.50000000e-02, 6.40000000e+00, 1.00000000e-01]
    q_xy = params[0] 
    q_wl = params[1] 
    q_v = params[2] 
    q_ori = params[3] 
    ha = params[4] 
    q_a= params[5] 
    q_ori_dot=params[6] 

    Q[0,0] = q_xy # x
    Q[1,1] = q_xy # y
    Q[2,2] = 0.0000000001 # z
    Q[3,3] = q_ori
    Q[4,4] = q_wl # x_size
    Q[5,5] = q_wl # y_size
    Q[6,6] = 0.0000000001 
    Q[7,7] = q_v # v_x
    Q[8,8] = q_v # v_y
    Q[9,9] = 0.0000000001 # v_z should be zero # TODO check that the order of Q is correct
    Q[10,10] = q_a # a_x
    Q[11,11] = q_a # a_y
    Q[12,12] = 0.0000000001 # a_z
    Q[13,13] = q_ori_dot # phi_dot
    q_params = "_xy" + str(q_xy) + "_ori" + str(q_ori) + "_wx" + str(q_wl) + "_ly" + str(q_wl) + "_v" +  str(q_v)
    '''
    
    
    
    qv=0.01953125
    qp=0.1
    q_params="qv_"+str(qv)
    if yl_utils.MOTION_MODEL == "CV":
      Q=yl_utils.get_CV_Q(qv,0.05)
    elif yl_utils.MOTION_MODEL == "CA":
        Q=yl_utils.get_CA_Q(qv,0.05)
    elif yl_utils.MOTION_MODEL == "CYRA":
        Q=yl_utils.get_CYRA_Q(qv,qp,0.05)
    else:
      print ("unknown motion model")
      raise ValueError
      
    R= np.identity(yl_utils.MEAS_SIZE)
    r_xy = 1. 
    r_ori = 0.06
    r_wl = 0.0390625
    
    # tracker_json_outfile = "/media/yl/downloads/tracker_results/set_8/"+yl_utils.MOTION_MODEL+"_state_10" + tracker_params +"_Q_"+ q_params + ".json"
    tracker_json_outfile = "/media/yl/downloads/tracker_results/set_8/newfp_cyra_state" + tracker_params +"_Q"+ q_params + ".json"
    start_tick = time.time()

    get_tracker_json(pixor_json_name=pixor_json_name, pixor_stats_json=None, tracker_json_outfile=tracker_json_outfile, 
      fused_pose_json=fused_pose_json, max_age=max_age,min_hits=min_hits,hung_thresh=hung_thresh, Q=Q, R=R)

    end_tick = time.time()

    print "Done writing to", tracker_json_outfile

    print "time taken: ", end_tick - start_tick
      
      

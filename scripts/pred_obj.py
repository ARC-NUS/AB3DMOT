#/usr/bin/env python

from filterpy.kalman import KalmanFilter
import collections
import yl_utils as yl
from docutils.io import InputError
import numpy as np


SW_COUNT = 5
PRED_MOTION_MOD = "CYRA"
PRED_STATE_SIZE = yl.STATE_SIZE
PRED_MEAS_SIZE = yl.MEAS_SIZE

obj_id_list=[]
pred_delta_t=0.5

# @var x = x position in utm
# TODO convert to UTM for prediction to use
class trk_pt():
  def __init__(self,start_time,init_state=None,kf_x=None):
    self.t= start_time
    if init_state is not None:
      self.x = init_state["geometry"]["position"]['x']
      self.y= init_state["geometry"]["position"]['y']
      self.ori= init_state["geometry"]['rotation']['z']
      self.w= init_state["geometry"]['dimensions']['x']
      self.l= init_state["geometry"]['dimensions']['y']
    elif kf_x is not None:
      self.x = init_state["geometry"]["position"]['x']
      self.y= init_state["geometry"]["position"]['y']
      self.ori= init_state["geometry"]['rotation']['z']
      self.w= init_state["geometry"]['dimensions']['x']
      self.l= init_state["geometry"]['dimensions']['y']
    else:
      print "trk_pt init: either init_state or kf_x must be defined"
      raise InputError
    
  def __str__(self):
    return " trk_pt:\n  x: "+ str(self.x)+\
            "\n  y: "+ str(self.y)+  \
            "\n  heading: "+ str(self.ori)+ \
            "\n  width (x): "+ str(self.w)+ \
            "\n  length (y): "+ str(self.l)+ \
            "\n  t: "+ str(self.t)
  def get_x(self):
    return np.array([self.x, self.y, 0.0, self.ori,
                     self.w, self.l, 1.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0])
  def get_z(self):
    return np.array([self.x, self.y, 0.0, self.ori,
                     self.w, self.l, 1.0])

'''
@param init_state: the array from the labels json
'''
class Pred_obj():
  
  def __init__(self,init_state,start_time,R,P,q_YR,q_A):
    self.id = init_state['classId']
    self.class_type = init_state['className']
    
    self.past_traj = []
    tmp=trk_pt(start_time=start_time, init_state=init_state)
    self.past_traj.append(tmp)
    
    obj_id_list.append(self.id)
    
    # KF for obj 
    self.kf = KalmanFilter(dim_x=yl.STATE_SIZE, dim_z=yl.STATE_SIZE) 
    if PRED_MOTION_MOD == "CYRA":
      self.kf.F = yl.get_CYRA_F(pred_delta_t) 
      self.kf.R = R
    else:
      print "unknown motion model:  ", PRED_MOTION_MOD
      raise TypeError
      
    self.kf.H = np.zeros((PRED_MEAS_SIZE,PRED_STATE_SIZE))
    for i in range(PRED_MEAS_SIZE):
      self.kf.H[i][i]=1.  
    self.kf.Q = None
    
    self.kf.P = P
    self.kf.x = self.past_traj[0].get_x()
    self.time_last_updated = start_time
    
    self.q_A=q_A
    self.q_YR=q_YR

  def __str__(self):
    tmp_str =  "Pred_obj:\n id: "+ str(self.id)+\
            "\n class_type: "+ self.class_type+\
            "\n past_traj:\n"
    for trk_pt_i in self.past_traj:
      tmp_str+=trk_pt_i.__str__()
      tmp_str+='\n'
    return tmp_str
            
  def update(self, new_state, curr_time):
    tmp=trk_pt(start_time=curr_time,init_state=new_state)
    self.past_traj.append(tmp)
    
    # TODO: update KF
    if PRED_MOTION_MOD == "CYRA":
      # update KF
      delta_t = (curr_time-self.time_last_updated) * 0.05 #TODO: only for 20Hz dataset
      self.kf.predict(F=get_CYRA_F(delta_t), Q=get_CYRA_Q(self.q_A, self.q_YR, delta_t)) 
      self.kf.update(tmp.get_z()) 
      self.time_last_updated = curr_time
    else:
      print "unknown motion model:  ", PRED_MOTION_MOD
      raise TypeError
    pass
    
  def predict(self, steps=3, delta_t=None, period=None): 
    if PRED_MOTION_MOD == "CYRA":
      self.predict_CYRA(steps, delta_t, period)
    else:
      print "unknown motion model:  ", PRED_MOTION_MOD
      raise TypeError
  
  def predict_CYRA(self, steps=3, delta_t=None, period=None):
    prediction = []
    if len(self.past_traj) >= 3:
      print "track too short, expected at least 3 previously known locations but only has ", \
            len(self.past_traj)
      return None
    else:
      # estimate const. acceleration
      for i in range(1): 
        if delta_t is None and period is None:
          
          tmp_Q=yl.get_CYRA_Q(self.q_A, self.q_YR, pred_delta_t)
          self.kf.Q = tmp_Q
          print "before: ", self.kf
          self.kf.predict(Q=tmp_Q)
          print "after: ", self.kf.x
          tmp = trk_pt(start_time=curr_time,init_state=new_state)
        # TODO handle else
      
      pass
    return prediction
    
  

if __name__ == '__main__':
  label_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/labels.old/Set_8_annotations.json"
  output_pred_json ="/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/prediction.json"
  
  fp_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/fused_pose/fused_pose_new.json"
  
  init_state = {}
  init_state['classId']=1
  init_state['className']="car_1"
  init_state['geometry']={}
  init_state['geometry']["position"]={}
  init_state['geometry']["rotation"]={}
  init_state['geometry']["dimensions"]={}
  init_state["geometry"]["position"]['x']=1.
  init_state["geometry"]["position"]['y']=2.
  init_state["geometry"]['rotation']['z']=3.
  init_state["geometry"]['dimensions']['x']=4.
  init_state["geometry"]['dimensions']['y']=5.
  
#   print init_state
  print Pred_obj(init_state,10.,np.eye(PRED_MEAS_SIZE),np.eye(PRED_STATE_SIZE),1.0,1.0)
   
#/usr/bin/env python

from filterpy.kalman import KalmanFilter, predict
import collections
import yl_utils as yl
from docutils.io import InputError
import numpy as np
from copy import deepcopy

# SW_COUNT = 5
# PRED_MOTION_MOD = "CYRA"
# PRED_MOTION_MOD = "CV"
PRED_MOTION_MOD = "CA"
PRED_STATE_SIZE = yl.STATE_SIZE
PRED_MEAS_SIZE = yl.MEAS_SIZE

obj_id_list=[]
pred_delta_t=0.5# in seconds
pred_steps = 17 # the first one will be the curr obj in the KF of the pred
COUNT_T=0.05 # one count in dataset is equivalent to 0.05s (for data)
label_count=0.5 # time distance bet each label in sec
MAX_STALE = 1.5 # if time since last up date is more than this then the object will b dropped from pred list. since for the labels, all obj must be labelled, we keep this val low
MIN_AGE = 3 # num of hits before start pred

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
      self.v_x = 0
      self.v_y = 0
      self.a_x = 0
      self.a_y = 0
    elif kf_x is not None:
      self.x = kf_x[0]
      self.y= kf_x[1]
      self.ori= kf_x[3]
      self.w= kf_x[4]
      self.l= kf_x[5]
      self.v_x = kf_x[7]
      self.v_y = kf_x[8]
      self.a_x = kf_x[10]
      self.a_y = kf_x[11]
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
                     self.w, self.l, 1.0]).reshape(PRED_MEAS_SIZE,1)

  def dict(self):
    tmp_dict = {"x": self.x, 
                "y": self.y,
                "w": self.w,
                "l": self.l,
                "v_x": self.v_x,
                "v_y": self.v_y,
                "a_x": self.a_x,
                "a_y": self.a_y,
                "heading": self.ori}
    return tmp_dict


'''
@param init_state: the array from the labels json
'''
class Pred_obj():
  
  def __init__(self,init_state,start_time,R,P,q_YR,q_A):
    self.id = init_state['classId']
    start_time*=COUNT_T
    self.class_type = init_state['className']
    
    self.past_traj = []
    tmp=trk_pt(start_time=start_time, init_state=init_state)
    self.past_traj.append(tmp)
    
    obj_id_list.append(self.id)
    
    # KF for obj 
    self.kf = KalmanFilter(dim_x=PRED_STATE_SIZE, dim_z=PRED_MEAS_SIZE) 
    if PRED_MOTION_MOD == "CYRA":
      self.kf.F = yl.get_CYRA_F(pred_delta_t) 
      self.kf.R = R
    elif PRED_MOTION_MOD == "CV":
      self.kf.F = yl.get_CV_F(pred_delta_t) 
      self.kf.R = R
    elif PRED_MOTION_MOD == "CA":
      self.kf.F = yl.get_CA_F(pred_delta_t) 
      self.kf.R = R
    else:
      print "unknown motion model:  ", PRED_MOTION_MOD
      raise TypeError
      
    self.kf.H = np.zeros((PRED_MEAS_SIZE,PRED_STATE_SIZE))
    for i in range(PRED_MEAS_SIZE):
      self.kf.H[i][i]=1.  
    
    self.q_A=q_A
    self.q_YR=q_YR
    self.kf.Q = None
    
    self.kf.P = P
    self.kf.x = self.past_traj[0].get_x()
    self.time_last_updated = start_time
    self.last_meas = None
    
  
  def check_stale(self,curr_timestep):
    curr_timestep *= COUNT_T
    is_stale = curr_timestep - self.time_last_updated > MAX_STALE
    if is_stale:
      self.suicide()
    return is_stale
    
  def suicide(self):
    obj_id_list.remove(self.id)
    return True

  def __str__(self):
    tmp_str =  "Pred_obj:\n id: "+ str(self.id)+\
            "\n class_type: "+ self.class_type+\
            "\n past_traj:\n"
    for trk_pt_i in self.past_traj:
      tmp_str+=trk_pt_i.__str__()
      tmp_str+='\n'
    return tmp_str
            
  def update(self, new_state, curr_time):
    curr_time *= COUNT_T
    tmp=trk_pt(start_time=curr_time,init_state=new_state)
    self.past_traj.append(tmp)
    
    # TODO: update KF
    if PRED_MOTION_MOD == "CYRA":
      # update KF
      delta_t = (curr_time-self.time_last_updated) 
      F = yl.get_CYRA_F(delta_t)
      Q = yl.get_CYRA_Q(self.q_A, self.q_YR, delta_t)
#       print "F,Q", F,Q
      self.kf.predict(F=F, Q=Q) 
      self.kf.update(tmp.get_z()) 
      self.time_last_updated = curr_time
    elif PRED_MOTION_MOD == "CV":
      # update KF
      delta_t = (curr_time-self.time_last_updated) 
      F = yl.get_CV_F(delta_t)
      Q = yl.get_CV_Q(self.q_A, delta_t)
#       print "F,Q\n", F,Q
      self.kf.predict(F=F, Q=Q) 
      self.kf.update(tmp.get_z()) 
      self.time_last_updated = curr_time
    elif PRED_MOTION_MOD == "CA":
      # update KF
      delta_t = (curr_time-self.time_last_updated) 
      F = yl.get_CA_F(delta_t)
      Q = yl.get_CA_Q(self.q_A, delta_t)
#       print "F,Q\n", F,Q
      self.kf.predict(F=F, Q=Q) 
      self.kf.update(tmp.get_z()) 
      self.time_last_updated = curr_time
    else:
      print "unknown motion model:  ", PRED_MOTION_MOD
      raise TypeError
    self.last_meas = tmp.get_x()
    pass
    
  def predict(self, curr_time, forced_state=False): 
    curr_time *= COUNT_T
    pred=self.predict_KF(curr_time, pred_steps,forced_state)
    return pred
    


  def predict_KF(self, curr_time, steps,forced_state):
    prediction = []
    # print 'predictionting for obj', self.id
    if len(self.past_traj) < MIN_AGE:
#       print "track too short, expected at least 3 previously known locations but only has ", \
#             len(self.past_traj)
      return None
    else:
      if forced_state:
        if curr_time-self.time_last_updated == 0:
          self.kf.x[0] = self.last_meas[0]
          self.kf.x[1] = self.last_meas[1]
          self.kf.x[3] = self.last_meas[3]
          self.kf.x[4] = self.last_meas[4]
          self.kf.x[5] = self.last_meas[5]
        else:
#           print "obj was not updated so cannot force fit", curr_time, self.id
          pass
            
      for i in range(steps): 
        t = pred_delta_t*(i) + curr_time-self.time_last_updated

        if PRED_MOTION_MOD == "CYRA":
          tmp_Q=yl.get_CYRA_Q(self.q_A, self.q_YR, t)
          tmp_F=yl.get_CYRA_F(t)  
        elif PRED_MOTION_MOD == "CV":
          tmp_Q=yl.get_CV_Q(self.q_A, t)
          tmp_F=yl.get_CV_F(t)
        elif PRED_MOTION_MOD == "CA":
          tmp_Q=yl.get_CA_Q(self.q_A, t)
          tmp_F=yl.get_CA_F(t) 
        else:
          print "unknown motion model:  ", PRED_MOTION_MOD
          raise TypeError
          
        x_p, P_p = predict(self.kf.x, self.kf.P, tmp_F, tmp_Q)

        tmp = trk_pt(start_time=pred_delta_t*(i+1),kf_x=x_p) # NOTE: any time update is to be done pred must be redone or else it will use the wrong prior

        prediction.append(tmp)
          
    # TODO handle else
    # TODO handel if inverse fails
      
    return prediction


if __name__ == '__main__':
  label_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/labels.old/Set_8_annotations.json"
  fp_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/fused_pose/fused_pose_new.json"
  output_pred_json ="/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/prediction.json"
  
  '''
  label_json='/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_annotations.json'
  output_pred_json = "/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pred_out.json"
  fp_json = "/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/fused_pose/fused_pose.json"
  '''
  
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
   

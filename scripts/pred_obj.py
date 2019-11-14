#/usr/bin/env python

from filterpy.kalman import KalmanFilter
import collections


SW_COUNT = 5

class trk():
  def __init__(self,init_state,t_):
    self.x = init_state[0]
    self.y= init_state[1]
    self.ori= init_state[2]
    self.w= init_state[3]
    self.l= init_state[4] # FIXME check the correct thingy
    self.t= t_

'''
@param init_state: the array from the labels json
'''
class pred_obj():
  def __init__(self,init_state,start_time):
    id = init_state['classId']
    class_type = init_state['className']
    past_traj = trk(init_state["geometry"], ) 


if __name__ == '__main__':
  label_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/labels.old/Set_8_annotations.json"
  output_pred_json ="/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/prediction.json"
  
  fp_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/fused_pose/fused_pose_new.json" 
#/usr/bin/env python

import yl_utils as yl
import json
from pred_obj import Pred_obj
import pred_obj as po
import numpy as np

def get_pred_json(label_json,output_pred_json,fused_pose_json,R,P,q_YR,q_A):
  pred_obj_list=[]

  with open(label_json) as json_file:
    with open(fused_pose_json) as fp_json:
      labels_data = json.load(json_file, encoding="utf-8")
      fp_data = json.load(fp_json, encoding="utf-8") # TODO use fp to do tf
      
      for index, labels in enumerate(labels_data): # for each pcd/timestep labelled
        curr_timestep = labels['name']
        curr_timestep = int(curr_timestep.split('.')[0])

        
        for obj in labels['annotations']:    
          obj_idx = 0
          for existing_obj_id in po.obj_id_list:
            if obj['classId'] == existing_obj_id:
              break
            else:
              obj_idx+=1
          if obj_idx == len(po.obj_id_list): 
            # create new object
            temp = Pred_obj(init_state=obj,start_time=curr_timestep, R=R,P=P,q_YR=q_YR,q_A=q_A)
            # print temp 
            pred_obj_list.append(temp)
          else:
            pred_obj_list[obj_idx].update(obj, curr_timestep)
            # TODO: convert to UTM

        # for each track, do the prediction
        for pred_i, p_o in enumerate(pred_obj_list):
          prediction = p_o.predict(curr_timestep) # TODO: switch model types
          if prediction is not None:
            print po.obj_id_list[pred_i], curr_timestep
            for tmp_pred in prediction:
              print tmp_pred

      # for p_o in pred_obj_list:
      #   print (p_o)
      

      


if __name__ == '__main__':
  
  label_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/labels.old/Set_8_annotations.json"
  output_pred_json ="/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/prediction.json"
  
  fp_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_8/fused_pose/fused_pose_new.json" 
  '''

  label_json='/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_annotations.json'
  output_pred_json = "/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pred_out.json"
  fp_json = "/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/fused_pose/fused_pose.json"
  '''


  R=np.eye(po.PRED_MEAS_SIZE)
  P=np.eye(po.PRED_STATE_SIZE)
  po.pred_delta_t=0.5
  q_YR=1.
  q_A=1.
  get_pred_json(label_json=label_json,output_pred_json=output_pred_json,fused_pose_json=fp_json,R=R,P=P,q_YR=q_YR,q_A=q_A)

  print "done"
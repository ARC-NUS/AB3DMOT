#/usr/bin/env python

import yl_utils as yl


def get_pred_json(label_json,output_pred_json):
  with open(label_json) as json_file:
    with open(fused_pose_json) as fp_json:
      labels_data = json.load(json_file, encoding="utf-8")
      fp_data = json.load(fp_json, encoding="utf-8") # TODO use fp to do tf
      
      for index, labels in enumerate(labels_data): # for each pcd/timestep labelled
        labels['name']
      
      


if __name__ == '__main__':
  label_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/labels.old/Set_8_annotations.json"
  output_pred_json ="/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/prediction.json"
  
  fp_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/fused_pose/fused_pose_new.json" 
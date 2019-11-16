#/usr/bin/env python


'''
@ summary scipt to check pred fr labels. since the assoc is alr done, we dont need to do association so its easier(?)

'''

import json
import math
from pred_obj import pred_delta_t, pred_steps, COUNT_T

def get_ADE(pred_json, labels_json):
  

  with open(labels_json) as json_file:
    with open(pred_json) as p_json:
      labels_data = json.load(json_file, encoding="utf-8")
      p_data = json.load(p_json, encoding="utf-8")

      ADE=[0.] * pred_steps
      ADE_count = [0.] * pred_steps

      for p_i, p_t in enumerate(p_data): # for each timestep in the p_data
        if p_t["curr_time"] != labels_data[p_i]['name']:
          print "error: label n pred json doesnt match. pred time: ", p_t['curr_time'], " label time: ", labels_data[p_i]['name']
          return None ## TODO handle for mismatch cases
        else:
          for obj in p_t['object_pred']:
            for t_i ,traj in enumerate(obj['traj']):
              # if it tries to look into the future which is beyond the labels, stop
              if p_i+int(t_i*pred_delta_t/COUNT_T) >= len(labels_data):
                break
              # get labels list for the matching pred
              for l_ in labels_data[p_i+int(t_i*pred_delta_t/0.05)]['annotations']:
                if l_['classId'] == obj["obj_id"]:
                  # print l_, obj["obj_id"]
                  ADE[t_i] +=(dist(l_,traj))
                  ADE_count[t_i]+=1
                  break
                # else:
                #   print 'label not found for pred obj pred time: ', p_t['curr_time'], " label time: ", labels_data[p_i]['name'], "obj:", obj["obj_id"]
  for i in range(len(ADE)):
    ADE[i] /= ADE_count[i]


  return ADE

def dist(label_obj, pred_obj):
  x = label_obj['geometry']['position']['x'] - pred_obj['x']
  y = label_obj['geometry']['position']['x'] - pred_obj['y']
  ori = label_obj['geometry']['rotation']['z'] - pred_obj['heading']

  # d = math.sqrt(x ** 2. + y ** 2. + ori ** 2.) # FIXME think if should scale ori
  d = math.sqrt(x ** 2. + y ** 2.) # FIXME think if should scale ori
  return d


if __name__ == '__main__':

  labels_json='/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_annotations.json'
  pred_json = "/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pred_out_0.1.json"  

  ADE=get_ADE(pred_json, labels_json)

  print ADE
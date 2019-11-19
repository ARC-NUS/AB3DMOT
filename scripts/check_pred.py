#/usr/bin/env python


'''
@ summary scipt to check pred fr labels. since the assoc is alr done, we dont need to do association so its easier(?)
can also gen image using is_write boolean
'''

import json
import math
from pred_obj import pred_delta_t, pred_steps, COUNT_T, label_count
import cv2
import numpy as np
import operator

is_write = True #
img_h = 800
img_w=800
scale = 10.0 # 1 px is to x cm

def get_ADE(pred_json, labels_json, img_path=None):
  with open(labels_json) as json_file:
    with open(pred_json) as p_json:
      labels_data = json.load(json_file, encoding="utf-8")
      p_data = json.load(p_json, encoding="utf-8")

      ADE=[0.] * pred_steps
      ADE_count = [0.] * pred_steps

      for p_i, p_t in enumerate(p_data): # for each timestep in the p_data
        img =  np.zeros((img_h,img_w,3), np.uint8)
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
              for l_ in labels_data[p_i+int(t_i*pred_delta_t/label_count)]['annotations']:
                if l_['classId'] == obj["obj_id"]:
                  # print l_, obj["obj_id"]
                  ADE[t_i] +=(dist(l_,traj))
                  ADE_count[t_i]+=1

                  if is_write:
                    img = draw(img,l_,traj,t_i)
                  break
                # else:
                #   print 'label not found for pred obj pred time: ', p_t['curr_time'], " label time: ", labels_data[p_i]['name'], "obj:", obj["obj_id"]

        if is_write:
          im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          cv2.imwrite(img_path+str(p_i)+".jpg", im_rgb)
  for i in range(len(ADE)):
    try:
      ADE[i] /= ADE_count[i]
    except ZeroDivisionError:
      ADE[i]=-float('inf')


  return ADE

def get_vertices(w,b,x_c,y_c,theta, img_h, img_w, scale):
  pts = np.array([[]], dtype=int)
  
  ptx = x_c + w /2.0*math.cos(theta) - b/2.0*math.sin(theta)
  pty = y_c + w /2.0*math.sin(theta) + b/2.0*math.cos(theta)

  ptx = int(round(1.0*ptx/scale + img_w/2.0)) 
  pty = img_h-int(round(1.0*pty/scale + img_h/2.0))
  pts = np.append(pts, [ptx, pty])

  ptx = x_c - w /2.0*math.cos(theta) - b/2.0*math.sin(theta)
  pty = y_c - w /2.0*math.sin(theta) + b/2.0*math.cos(theta)

  ptx = int(round(1.0*ptx/scale + img_w/2.0)) 
  pty = img_h-int(round(1.0*pty/scale + img_h/2.0))
  pts = np.vstack((pts, [ptx, pty]))

  ptx = x_c - w /2.0*math.cos(theta) + b/2.0*math.sin(theta)
  pty = y_c - w /2.0*math.sin(theta) - b/2.0*math.cos(theta)

  ptx = int(round(1.0*ptx/scale + img_w/2.0)) 
  pty = img_h-int(round(1.0*pty/scale + img_h/2.0))
  pts = np.vstack((pts, [ptx, pty]))            

  ptx = x_c + w /2.0*math.cos(theta) + b/2.0*math.sin(theta)
  pty = y_c + w /2.0*math.sin(theta) - b/2.0*math.cos(theta)

  ptx = int(round(1.0*ptx/scale + img_w/2.0)) 
  pty = img_h-int(round(1.0*pty/scale + img_h/2.0))
  pts = np.vstack((pts, [ptx, pty]))        
  return pts

def draw_border(img, pts, clr, thiccness=2):
  for i in range(len(pts) -1):
#         print pts[i]
    cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), clr, thiccness)
  cv2.line(img, tuple(pts[0]), tuple(pts[len(pts)-1]), clr, thiccness)
  return img


def draw(img,label_obj,pred_obj,iter):
  '''
  # FIXME only works for steps that are less than 6
  l_clr_dict = { 0: (255,20,147), 
                 1: (255,50,147), # FF00FF
                 2: (255,100,147), # FF33FF
                 3: (255,150,147), # FF66FF
                 4: (255,200,147), # FF99FF
                 5: (255,250,147)} 
  p_clr_dict = { 0: (00,255,255), # 00FFFF
                 1: (33,255,255), # 33FFFF
                 2: (66,255,255), # 66FFFF
                 3: (99,255,255), # 99FFFF
                 4: (204,255,255), # CCFFFF
                 5: (210,255,255)} # D2FFFF
  '''

  frac = iter/16.
  l_1 =  np.array([255,20,147])
  l_2 =  np.array([255,255,10])
  p_1 = np.array([0,255,255])
  p_2 = np.array([10,188,158])

  l_np = (l_2 - l_1) * frac + l_1
  l_clr = tuple(l_np.astype(int))
  p_np = (p_2 - p_1) * frac + p_1
  p_clr = tuple(p_np.astype(int))

#   print frac,l_clr, p_clr

  # draw label box
  w = float(label_obj['geometry']['dimensions']['x'])
  b = float(label_obj['geometry']['dimensions']['y'])
  x_c = float(label_obj['geometry']['position']['x'])
  y_c = float(label_obj['geometry']['position']['y'])
  theta = float(label_obj['geometry']['rotation']['z'])
  pts = get_vertices(w,b,x_c,y_c,theta,img_h,img_w,scale/100.)
  img = draw_border(img, pts, l_clr,4+(5-iter/2))

  # draw pixor labels
  x_c = float(pred_obj['x'])
  y_c = float(pred_obj['y'])
  theta = float(pred_obj['heading'])
  pts = get_vertices(w,b,x_c,y_c,theta,img_h,img_w,scale/100.)
  img = draw_border(img, pts, p_clr, 2+(5-iter/2))

  return img


def dist(label_obj, pred_obj):
  x = label_obj['geometry']['position']['x'] - pred_obj['x']
  y = label_obj['geometry']['position']['y'] - pred_obj['y']
  ori = label_obj['geometry']['rotation']['z'] - pred_obj['heading']

  # d = math.sqrt(x ** 2. + y ** 2. + ori ** 2.) # FIXME think if should scale ori
  d = math.sqrt(x ** 2. + y ** 2.) # FIXME think if should scale ori
  return d


if __name__ == '__main__':
  '''
  labels_json='/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_annotations.json'
  pred_json = "/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pred_out_0.5.json"  
  img_path ="/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/img_pred/"  
  '''

  labels_json='/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_Correct_annotations.json'
  pred_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pred_out_8.json"  
  img_path ="/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/img_pred/"  


  ADE=get_ADE(pred_json, labels_json,img_path)

  print ADE

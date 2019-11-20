#/usr/bin/env python


'''
@ summary scipt to check pred fr labels. since the assoc is alr done, we dont need to do association so its easier(?)
can also gen image using is_write boolean
'''

import json
import math
from pred_obj import pred_delta_t, pred_steps, COUNT_T, label_count,PRED_MEAS_SIZE,PRED_STATE_SIZE
import cv2
import numpy as np
import operator
import yl_utils as yl
from predictor_wt_labels import get_pred_json
from os import listdir, walk, pardir, makedirs, errno


is_write = False 
img_h = 2000
img_w = 2000
scale = 10.0 # 1 px is to x cm

def get_ADE(pred_json=None, labels_json=None, img_path=None, pred_list=None):
  
  if pred_json is not None:
    p_json = open(pred_json)
    p_data = json.load(p_json, encoding="utf-8")
  elif pred_list is not None:
    p_data = pred_list
  else:
    print "error, get_ADE requires either json file or list obj as input"
    return None
  
  with open(labels_json) as json_file:
    labels_data = json.load(json_file, encoding="utf-8")

    ADE=[0.] * pred_steps
    ADE_count = [0.] * pred_steps

    for p_i, p_t in enumerate(p_data): # for each timestep in the p_data
      img =  np.zeros((img_h,img_w,3), np.uint8)
      
      # print prev label
      if p_i-1 >= 0 and is_write:
        for curr_label in labels_data[p_i-1]['annotations']:
          img = draw_label(img, curr_label, -1)
                
      if p_t["curr_time"] != labels_data[p_i]['name']:
        print "error: label n pred json doesnt match. pred time: ", p_t['curr_time'], " label time: ", labels_data[p_i]['name']
        return None ## TODO handle for mismatch cases
      else:
        for obj in p_t['object_pred']:
          for t_i ,traj in enumerate(obj['traj']):
            # if it tries to look into the future which is beyond the labels, stop
            label_id = p_i+int(t_i*pred_delta_t/label_count)

            if label_id >= len(labels_data):
#               print label_id, " is out of label bounds"
              pass
            else:
              # get labels list for the matching pred
              for l_ in labels_data[label_id]['annotations']:
                if l_['classId'] == obj["obj_id"]:
                  # print l_, obj["obj_id"]
                  ADE[t_i] +=(dist(l_,traj))
                  ADE_count[t_i]+=1
  
                  if is_write:
                    img = draw_label(img, l_, t_i)
                  break
                # else:
                #   print 'label not found for pred obj pred time: ', p_t['curr_time'], " label time: ", labels_data[p_i]['name'], "obj:", obj["obj_id"]
            
            # draw all pred
            if is_write:
#               print traj
              img = draw_pred(img, traj, t_i)
              
      if is_write:
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path+str(p_i)+".jpg", im_rgb)
  return np.array(ADE), np.array(ADE_count)
  
  
  
def get_multi_ADE(parent_folder,R,P,q_YR,q_A):
   
  high_set_v, labels_paths_v = yl.get_dirs(parent_folder,"pixor_train")
  t_high_set_v, t_labels_paths_v = yl.get_dirs(parent_folder,"pixor_test")
  high_set_v += t_high_set_v
  labels_paths_v += t_labels_paths_v
  t_high_set_v, t_labels_paths_v = yl.get_dirs(parent_folder,"pixor_test_hard")
  high_set_v += t_high_set_v
  labels_paths_v += t_labels_paths_v

#   high_set_v, labels_paths_v = yl.get_dirs(parent_folder,"track_test")
  
#   high_set_v, labels_paths_v = yl.get_dirs(parent_folder,"pixor_train")
  
  ADE=np.zeros(pred_steps)
  ADE_count = np.zeros(pred_steps)
    
  for l_i, label in enumerate(labels_paths_v):
    fp_json = high_set_v[l_i]+"/fused_pose/fused_pose_new.json"
#     print "working on file: ",label
    img_dir = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/img_pred/" + str(l_i)
    if is_write:
      try:
        makedirs(img_dir)
      except OSError, e:
        if e.errno != errno.EEXIST:
          raise
        else:
          print img_dir, "dir exists. overwritting file "
      img_path = img_dir+"/img_"
    else:
      img_path = None
    pred_list=get_pred_json(label_json=label,output_pred_json=None,
                  fused_pose_json=fp_json,R=R,P=P,q_YR=q_YR,q_A=q_A)  
    tmp_ADE, tmp_ADE_c = get_ADE(labels_json=label, 
                                 img_path=img_path,
                                 pred_list=pred_list)
    ADE += tmp_ADE
    ADE_count += tmp_ADE_c
      
  return ADE/ADE_count




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


def draw_label(img,label_obj,iter):
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

  frac = iter/8.
  l_1 =  np.array([255,20,147])
  l_2 =  np.array([255,255,10])
  

  l_np = (l_2 - l_1) * frac + l_1
  l_clr = tuple(l_np.astype(int))
  
  if iter < 0:
    l_clr = (125,10,70)

#   print frac,l_clr, p_clr

  # draw label box
  w = float(label_obj['geometry']['dimensions']['x'])
  b = float(label_obj['geometry']['dimensions']['y'])
  x_c = float(label_obj['geometry']['position']['x'])
  y_c = float(label_obj['geometry']['position']['y'])
  theta = float(label_obj['geometry']['rotation']['z'])
  pts = get_vertices(w,b,x_c,y_c,theta,img_h,img_w,scale/100.)
  img = draw_border(img, pts, l_clr,6+(5-iter/3))
  
  # draw iter
  cv2.putText(img,str(iter),\
              (int(round(img_w/2.0+(x_c*100./scale))), \
              img_h-int(round(img_h/2.0+y_c*100./scale)) + 30 ), \
              cv2.FONT_HERSHEY_SIMPLEX, 1, l_clr, \
              3, cv2.LINE_AA)

#   cv2.putText(img,str(x_c)[0:6],\
#               (int(round(img_w/2.0+(x_c*100./scale))) + 10, \
#               img_h-int(round(img_h/2.0+y_c*100./scale)) + 50 ), \
#               cv2.FONT_HERSHEY_SIMPLEX, 1, l_clr, \
#               3, cv2.LINE_AA)
#   cv2.putText(img,str(y_c)[0:6],\
#               (int(round(img_w/2.0+(x_c*100./scale))) + 10, \
#               img_h-int(round(img_h/2.0+y_c*100./scale)) + 80 ), \
#               cv2.FONT_HERSHEY_SIMPLEX, 1, l_clr, \
#               3, cv2.LINE_AA)
  return img

def draw_pred(img,pred_obj,iter):
  frac = iter/16.
  p_1 = np.array([0,255,255])
  p_2 = np.array([10,102,51])
  
  p_np = (p_2 - p_1) * frac + p_1
  p_clr = tuple(p_np.astype(int))
  
  # draw pred labels
  x_c = float(pred_obj['x'])
  y_c = float(pred_obj['y'])
  w = float(pred_obj['w'])
  b = pred_obj['l']
  theta = float(pred_obj['heading'])
  pts = get_vertices(w,b,x_c,y_c,theta,img_h,img_w,scale/100.)
  img = draw_border(img, pts, p_clr, 2+(5-iter/3))
  
  # draw iter
  if iter == 0:
    cv2.putText(img,str(iter),\
              (int(round(img_w/2.0+(x_c*100./scale))), \
              img_h-int(round(img_h/2.0+y_c*100./scale)) + 30 ), \
              cv2.FONT_HERSHEY_SIMPLEX, 1, p_clr, \
              3, cv2.LINE_AA)
              # draw vel
    cv2.putText(img,str(pred_obj['v_x'])[0:6],\
              (int(round(img_w/2.0+(x_c*100./scale))) + 10, \
              img_h-int(round(img_h/2.0+y_c*100./scale)) + 50 ), \
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), \
              3, cv2.LINE_AA)
    cv2.putText(img,str(pred_obj['a_x'])[0:6],\
              (int(round(img_w/2.0+(x_c*100./scale))) + 10, \
              img_h-int(round(img_h/2.0+y_c*100./scale)) + 80 ), \
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), \
              3, cv2.LINE_AA)
#   cv2.putText(img,str(x_c)[0:6],\
#             (int(round(img_w/2.0+(x_c*100./scale))) + 10, \
#             img_h-int(round(img_h/2.0+y_c*100./scale)) + 50 ), \
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), \
#             3, cv2.LINE_AA)
#   cv2.putText(img,str(y_c)[0:6],\
#             (int(round(img_w/2.0+(x_c*100./scale))) + 10, \
#             img_h-int(round(img_h/2.0+y_c*100./scale)) + 80 ), \
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), \
#             3, cv2.LINE_AA)
    
  return img


def dist(label_obj, pred_obj):
  x = label_obj['geometry']['position']['x'] - pred_obj['x']
  y = label_obj['geometry']['position']['y'] - pred_obj['y']
  ori = label_obj['geometry']['rotation']['z'] - pred_obj['heading']

  # d = math.sqrt(x ** 2. + y ** 2. + ori ** 2.) # FIXME think if should scale ori
  d = math.sqrt(x ** 2. + y ** 2.) # FIXME think if should scale ori
  return d



def test_single_jsons():
  '''
  labels_json='/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_annotations.json'
  pred_json = "/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pred_out_0.5.json"  
  img_path ="/home/yl/Downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/img_pred/"  
  '''

  labels_json='/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_Correct_annotations.json'
  pred_json = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pred_out_8_2.json"  
  img_path ="/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/img_pred/"  

  ADE,ADE_count=get_ADE(pred_json, labels_json,img_path)
  
  for i in range(len(ADE)):
    try:
      ADE[i] /= ADE_count[i]
    except ZeroDivisionError:
      ADE[i]=-float('inf')

  print ADE
  
  
def test_multi_jsons():
  parent_dir = "/media/yl/demo_ssd/raw_data"
#   R=np.eye(PRED_MEAS_SIZE)
#   P=np.eye(PRED_STATE_SIZE)
#   q_YR=2.
#   q_A=2.
#   params_v = [2.,   0.16, 2.5,  1.25] 
#   params_v = [1.00000000e+03, 5.12000000e-04, 8.88178420e-01, 1.12589991e+00] 

#   q_YR = params_v[0]
#   q_A = params_v[1]
#   R=np.eye(PRED_MEAS_SIZE) * params_v[2]
#   P=np.eye(PRED_STATE_SIZE) * params_v[3]
  
  # FIXME
  q_YR = 10. ** 10
  q_A = 10. ** 100
  R=np.eye(PRED_MEAS_SIZE) * (10**-10)
  
  P=np.eye(PRED_STATE_SIZE) * (10.**10.)
  unseen_p = 10. ** 100.
  P[7,7]=unseen_p
  P[8,8]=unseen_p
  P[10,10]=unseen_p
  P[11,11]=unseen_p
  P[13,13]=unseen_p
  
  ADE = get_multi_ADE(parent_dir,R,P,q_YR,q_A)
  print ADE
  
  
if __name__ == '__main__':
  np.set_printoptions(precision=3, linewidth=100000)
  test_multi_jsons()
#   test_single_jsons()


#!/usr/bin/env python
# coding: utf-8

# In[1]:

from IPython import get_ipython
get_ipython().magic(u'matplotlib')

# visualise data from json files
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


# In[2]:


# 2 hz labels
#data_path = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/"
data_path = "/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/"
#labels_json_path = data_path + "/log_low/set_7/no_qc/Set_7_Correct_annotations.json"
labels_json_path = data_path + "/log_low/set_1/labels/set1_annotations.json"

# 20hz pixor
#pixor_json_path = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs_tf_epoch_3_valloss_0.0093_2.json"
pixor_json_path = "../../raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_low/set_1/pixor_outputs_tf_epoch_49_valloss_0.0117.json"
#pixor_json_path = "../../raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_low/set_1/old pixor outputs/pixor_outputs.json"

# 20hz tracker
#tracker_json_path = "/media/yl/downloads/tracker_results/set_7/newfp_cyra_statemax_age=6,min_hits=3,hung_thresh=0.25_Qqv_10.0.json"
tracker_json_path = "person.json"

# output image folder. WARNING: IMAGE FOLDER MUST ALREADY EXIST
#img_path = "/home/yl/bus_ws/src/auto_bus/perception/ros_to_rawdata/files/test/"

# output image folder. WARNING: IMAGE FOLDER MUST ALREADY EXIST
img_path = "/home/wen/AB3DMOT/scripts/results/JI_Cetran_Set1/data/tracker_images/"

# In[3]:


def draw_border(img, pts, clr, thiccness=2):
    for i in range(len(pts) -1):
#         print pts[i]
        cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), clr, thiccness)
    cv2.line(img, tuple(pts[0]), tuple(pts[len(pts)-1]), clr, thiccness)
    return img


# In[4]:


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


# In[5]:


img_h = 500
img_w = 1000
scale = 20.0 # 1 px is to x cm
STOP_COUNT = 100

px_l_ratio = 1


# In[6]:


with open(pixor_json_path) as pixor_json_file:
    with open(labels_json_path) as labels_json_file:
        with open(tracker_json_path) as tracker_json_file:
            pixor_data = json.load(pixor_json_file, encoding="utf-8")
            labels_data = json.load(labels_json_file, encoding="utf-8")
            tracker_data = json.load(tracker_json_file, encoding="utf-8")
            px_i = 0 # for accounting for empty labels

            for i in range(0, STOP_COUNT):
                img = np.zeros((img_h,img_w,3), np.uint8)

                # plot bus ego
                cv2.rectangle(img,
                              (int(round(img_w/2.0-(3.5*100./scale))),
                               int(round(img_h/2.0+1.5*100./scale))),
                              (int(round(img_w/2.0+8.5*100./scale)),
                               int(round(img_h/2.0-1.5*100./scale))),
                              (255,0,0), -1)

                # plot pixor fov
                cv2.rectangle(img,
                              (int(round(img_w/2.0-(35.2*100./scale))),
                               int(round(img_h/2.0+20*100./scale))),
                              (int(round(img_w/2.0+35.2*100./scale)),
                               int(round(img_h/2.0-20*100./scale))),
                              (255,255,255), 1)


                # TODO: use names and count to check the corresponding data



                # PIXOR json --> yellow
                pxr_det = pixor_data[i*10+9]
                pxr_clr = (255,255,0)
                for det in pxr_det['objects']:
                    w = float(det['length']) 
                    b = float(det['width']) # width is in the y direction for Louis
                    x_c = float(det['centroid'][0])
                    y_c = float(det['centroid'][1])
                    theta = float(det['heading'])
                    pts = get_vertices(w,b,x_c,y_c,theta,img_h,img_w,scale/100.)
                    img = draw_border(img, pts, pxr_clr,thiccness=5)

                labels = labels_data[i]
                l_clr = (255,20,147)
                if labels['name'] != pxr_det['name']:
                    print "error mismatched label and pixor output pcd!"
                    print "pixor pcd: ", pxr_det['name']
                    print "label pcd: ", labels['name']
                    px_i +=1
                    continue

                for label in labels['annotations']:
                    w = float(label['geometry']['dimensions']['x'])
                    b = float(label['geometry']['dimensions']['y'])
                    x_c = float(label['geometry']['position']['x'])
                    y_c = float(label['geometry']['position']['y'])
                    theta = float(label['geometry']['rotation']['z'])
                    pts = get_vertices(w,b,x_c,y_c,theta,img_h,img_w,scale/100.)
                    if label['className'].find("truck") >= 0                         or label['className'].find("bus") >= 0                         or label['className'].find("car")>= 0                         or label['className'].find("van")>= 0:
                        img = draw_border(img, pts, l_clr)
                        cv2.putText(img,str(label['classId']),                                (int(round(img_w/2.0+(x_c*100./scale))),                                 img_h-int(round(img_h/2.0+y_c*100./scale)) + 30 ),                                 cv2.FONT_HERSHEY_SIMPLEX, 1, l_clr,                                 3, cv2.LINE_AA)
                    else:
#                                 print "label class:" ,label['className']
                        img = draw_border(img, pts, (100,100,100))
                        cv2.putText(img,str(label['classId']),                                (int(round(img_w/2.0+(x_c*100./scale))),                                 img_h-int(round(img_h/2.0+y_c*100./scale)) + 30 ),                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100),                                     3, cv2.LINE_AA)

                # tracker json --> blue
                frame = tracker_data[i*10]

                for obj in frame['objects']:
                    track_clr = (0,191,255)
                    if (len(obj) != 0):
                        w = float(obj['width'])
                        b = float(obj['length'])
                        x_c = float(obj['x'])
                        y_c = float(obj['y'])
                        theta = float(obj['yaw'])
                        pts = get_vertices(w,b,x_c,y_c,theta,img_h,img_w,scale/100.)
                        img = draw_border(img, pts, track_clr)
                        cv2.putText(img,str(obj['id']),                                    (int(round(img_w/2.0+(x_c*100./scale))),                                     img_h-int(round(img_h/2.0+y_c*100./scale)) -10),                                     cv2.FONT_HERSHEY_SIMPLEX, 1, track_clr ,                                     3, cv2.LINE_AA)

                im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_path+"/img_"+str(i)+".jpg", im_rgb)

print "Done"


# In[ ]:





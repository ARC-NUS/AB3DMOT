#!/usr/bin/env python
# coding: utf-8
## coordinate system conventions
#width is in y-direction (x is forward of the bus)
#35.2m front back,  -/+20m sideways from baselink
#The heading is with respect to the baselink's x 'in rad.

# In[1]:

from IPython import get_ipython
get_ipython().magic(u'matplotlib')

# visualise data from json files
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from datetime import date
import os

# In[2]:


# 2 hz labels
#data_path = "/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/"
data_path = "/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/"
#data_path = "/media/wen/ARC_SSD_2/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/"

#labels_json_path = data_path + "/log_low/set_7/no_qc/Set_7_Correct_annotations.json"
labels_json_path = data_path + "/log_high/set_1/labels/set1_annotations.json"

# 20hz pixor
#pixor_json_path = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs_tf_epoch_3_valloss_0.0093_2.json"
#pixor_json_path = "../../raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_low/set_1/pixor_outputs_tf_epoch_42_valloss_0.0112.json"
pixor_json_path = "../../raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/set_1/pixor_outputs_tf_epoch_23_valloss_0.0087.json"

#pixor_json_path = data_path + "/log_low/set_1/labels/set1_annotations.json"

# RADAR VALUES
radar_obstacles_path = "../../raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/set_1/radar_obstacles/radar_obstacles.json"

# 20hz tracker
#tracker_json_path = "/media/yl/downloads/tracker_results/set_7/newfp_cyra_statemax_age=6,min_hits=3,hung_thresh=0.25_Qqv_10.0.json"
tracker_json_path = "./results/JI_Cetran_Set1/SensorFusedTrackOutput_Set1_31_01_2020.json"

# 20hz tracker
#tracker_json_path = "/media/yl/downloads/tracker_results/set_7/newfp_cyra_statemax_age=6,min_hits=3,hung_thresh=0.25_Qqv_10.0.json"
tracker_json_path2 = "./results/JI_Cetran_Set1/yltracker2.json"

# output image folder. WARNING: IMAGE FOLDER MUST ALREADY EXIST
#img_path = "/home/yl/bus_ws/src/auto_bus/perception/ros_to_rawdata/files/test/"

today = date.today()
d1 = today.strftime("%d_%m_%Y")

# output image folder. WARNING: IMAGE FOLDER MUST ALREADY EXIST
img_path = "/home/wen/AB3DMOT/scripts/results/JI_Cetran_Set1/data/tracker_images/" + d1 + "/"

if not os.path.exists(img_path):
    os.makedirs(img_path)

# In[3]:


def draw_border(img, pts, clr, thiccness=2):
    for i in range(len(pts) -1):
        #print pts[i]
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

with open(radar_obstacles_path) as radar_obstacles_file:
    with open(pixor_json_path) as pixor_json_file:
        with open(labels_json_path) as labels_json_file:
            with open(tracker_json_path) as tracker_json_file:
                with open(tracker_json_path2) as tracker_json_file2:
                    pixor_data = json.load(pixor_json_file, encoding="utf-8")
                    labels_data = json.load(labels_json_file, encoding="utf-8")
                    tracker_data = json.load(tracker_json_file, encoding="utf-8")
                    tracker_data2 = json.load(tracker_json_file2, encoding="utf-8")
                    radar_data = json.load(radar_obstacles_file, encoding="utf-8")
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
                        #pxr_det = pixor_data[i]
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
                                cv2.putText(img,str(label['classId']), (int(round(img_w/2.0+(x_c*100./scale))), img_h-int(round(img_h/2.0+y_c*100./scale)) + 30 ), cv2.FONT_HERSHEY_SIMPLEX, 1, l_clr, 3, cv2.LINE_AA)
                            else:
        #                                 print "label class:" ,label['className']
                                img = draw_border(img, pts, (100,100,100))
                                cv2.putText(img,str(label['classId']),    (int(round(img_w/2.0+(x_c*100./scale))), img_h-int(round(img_h/2.0+y_c*100./scale)) + 30 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 3, cv2.LINE_AA)

                        # tracker 2 json --> green
                        frame = tracker_data2[i*10+9]
                        for obj in frame['objects']:
                            track_clr = (50, 205, 50)
                            if (len(obj) != 0):
                                w = float(obj['width'])
                                b = float(obj['length'])
                                x_c = float(obj['x'])
                                y_c = float(obj['y'])
                                theta = float(obj['yaw'])
                                pts = get_vertices(w,b,x_c,y_c,theta,img_h,img_w,scale/100.)
                                img = draw_border(img, pts, track_clr)
                                cv2.putText(img,str(obj['id']),(int(round(img_w/2.0+(x_c*100./scale))), img_h-int(round(img_h/2.0+y_c*100./scale)) -10), cv2.FONT_HERSHEY_SIMPLEX, 1, track_clr ,3, cv2.LINE_AA)

                        #tracker json --> blue
                        frame = tracker_data[i*10+9]
                        for obj in frame['objects']:
                            track_clr = (0,191,255)
                            if (len(obj) != 0):
                                w = float(obj['width'])
                                b = float(obj['length'])
                                x_c = float(obj['x'])
                                y_c = float(obj['y'])
                                theta = float(obj['yaw'])
                                pts = get_vertices(w, b, x_c, y_c, theta, img_h, img_w, scale / 100.)
                                img = draw_border(img, pts, track_clr)
                                cv2.putText(img, str(obj['id']), (int(round(img_w / 2.0 + (x_c * 100. / scale))),
                                                                  img_h - int(
                                                                      round(img_h / 2.0 + y_c * 100. / scale)) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, track_clr, 3, cv2.LINE_AA)
                        #TODO must add to the bus pose

                        # radar_Data = radar_data['radar'][i*10]
                        # for obj in radar_Data['front_esr_tracklist']:
                        #     track_clr = (100, 191, 255)
                        #     radius = 1
                        #     thickness = 1
                        #     if (len(obj) != 0):
                        #         range = float(obj['range'])
                        #         theta_r = float(obj['angle_centroid'])
                        #         theta_r = np.deg2rad(theta)
                        #         x_cr = int(round(range * np.cos(theta)))
                        #         y_cr = int(round(range * np.sin(theta)))
                        #         cv2.circle(img, (x_cr,y_cr), radius, track_clr, thickness)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale =0.7
                        thickness = 1


                        cv2.putText(img, "Tracker_(Lidar Only)", (50, 50), font, fontScale,
                                    (50, 205, 50), thickness)
                        cv2.putText(img, "Tracker_(Sensor Fusion)", (50, 70), font, fontScale, (0,191,255), thickness)
                        cv2.putText(img, "PIXOR++", (50, 90), font, fontScale, (255, 255, 0), thickness)
                        cv2.putText(img, "Labels", (50, 110), font, fontScale, (255, 20, 147), thickness)

                        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        #cv2.imwrite(img_path+"/img_"+str(i)+".jpg", im_rgb)
                        cv2.imwrite(img_path + "/img_" + labels['name'][1:5] + ".jpg", im_rgb)

print "Done"


# In[ ]:





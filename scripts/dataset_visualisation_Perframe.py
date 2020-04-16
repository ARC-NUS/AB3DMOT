#!/usr/bin/env python
# coding: utf-8
## coordinate system conventions
# width is in y-direction (x is forward of the bus)
# 35.2m front back,  -/+20m sideways from baselink
# The heading is with respect to the baselink's x 'in rad.

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
import glob
# In[2]:

today = date.today()
d1 = today.strftime("%Y_%m_%d")


basedir_total = ['/media/wen/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_2']
labels_total = ['/media/wen/demo_ssd/raw_data/eval_labels/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/set_2/']

print('Trying testcase 12566!! ')
i = 0  # to be the one with pedesterians
basedir = basedir_total[i]
print(basedir)
labels_json_path = glob.glob(labels_total[i] + "/*annotations.json")
labels_json_path= labels_json_path[0]
# Join various path components
pixor_json_path = basedir + '/pixor_outputs_mdl_tf_epoch_150_valloss_0.2106.json'
img_path = basedir + "/tracker_visualise/" + d1 + "pf2/"
tracker_json_path = "/home/wen/AB3DMOT/scripts/results/sensorfusion/checkSF.json"
tracker_json_path2 = "/media/wen/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_2/trackerresults_wCR_12566_2020_04_06.json"

pathIBEO = basedir + '/ecu_obj_list/ecu_obj_list.json'

if not os.path.exists(img_path):
    os.makedirs(img_path)


# In[3]:


def draw_border(img, pts, clr, thiccness=2):
    for i in range(len(pts) - 1):
        # print pts[i]
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), clr, thiccness)
    cv2.line(img, tuple(pts[0]), tuple(pts[len(pts) - 1]), clr, thiccness)
    return img


# In[4]:


def get_vertices(w, b, x_c, y_c, theta, img_h, img_w, scale):
    pts = np.array([[]], dtype=int)

    ptx = x_c + w / 2.0 * math.cos(theta) - b / 2.0 * math.sin(theta)
    pty = y_c + w / 2.0 * math.sin(theta) + b / 2.0 * math.cos(theta)

    ptx = int(round(1.0 * ptx / scale + img_w / 2.0))
    pty = img_h - int(round(1.0 * pty / scale + img_h / 2.0))
    pts = np.append(pts, [ptx, pty])

    ptx = x_c - w / 2.0 * math.cos(theta) - b / 2.0 * math.sin(theta)
    pty = y_c - w / 2.0 * math.sin(theta) + b / 2.0 * math.cos(theta)

    ptx = int(round(1.0 * ptx / scale + img_w / 2.0))
    pty = img_h - int(round(1.0 * pty / scale + img_h / 2.0))
    pts = np.vstack((pts, [ptx, pty]))

    ptx = x_c - w / 2.0 * math.cos(theta) + b / 2.0 * math.sin(theta)
    pty = y_c - w / 2.0 * math.sin(theta) - b / 2.0 * math.cos(theta)

    ptx = int(round(1.0 * ptx / scale + img_w / 2.0))
    pty = img_h - int(round(1.0 * pty / scale + img_h / 2.0))
    pts = np.vstack((pts, [ptx, pty]))

    ptx = x_c + w / 2.0 * math.cos(theta) + b / 2.0 * math.sin(theta)
    pty = y_c + w / 2.0 * math.sin(theta) - b / 2.0 * math.cos(theta)

    ptx = int(round(1.0 * ptx / scale + img_w / 2.0))
    pty = img_h - int(round(1.0 * pty / scale + img_h / 2.0))
    pts = np.vstack((pts, [ptx, pty]))
    return pts


# In[5]:


img_h = 500
img_w = 1000
scale = 20.0  # 1 px is to x cm
STOP_COUNT = 1000

px_l_ratio = 1

# In[6]:
fthick =2

with open(pathIBEO) as ibeo_json_file:
    with open(pixor_json_path) as pixor_json_file:
        with open(labels_json_path) as labels_json_file:
            with open(tracker_json_path) as tracker_json_file:
                with open(tracker_json_path2) as tracker_json_file2:
                    pixor_data = json.load(pixor_json_file, encoding="utf-8")
                    labels_data = json.load(labels_json_file, encoding="utf-8")
                    tracker_data = json.load(tracker_json_file, encoding="utf-8")
                    tracker_data2 = json.load(tracker_json_file2, encoding="utf-8")
                    ibeo_data = json.load(ibeo_json_file, encoding="utf-8")
                    ibeo_data = ibeo_data['ibeo_obj']
                    px_i = 0  # for accounting for empty labels

                    for i in range(0, STOP_COUNT):
                        img = np.zeros((img_h, img_w, 3), np.uint8)

                        # plot bus ego
                        cv2.rectangle(img,
                                      (int(round(img_w / 2.0 - (3.5 * 100. / scale))),
                                       int(round(img_h / 2.0 + 1.5 * 100. / scale))),
                                      (int(round(img_w / 2.0 + 8.5 * 100. / scale)),
                                       int(round(img_h / 2.0 - 1.5 * 100. / scale))),
                                      (255, 0, 0), -1)

                        # plot pixor fov
                        cv2.rectangle(img,
                                      (int(round(img_w / 2.0 - (35.2 * 100. / scale))),
                                       int(round(img_h / 2.0 + 20 * 100. / scale))),
                                      (int(round(img_w / 2.0 + 35.2 * 100. / scale)),
                                       int(round(img_h / 2.0 - 20 * 100. / scale))),
                                      (255, 255, 255), 1)

                        # TODO: use names and count to check the corresponding data

                        # PIXOR json --> yellow
                        pxr_det = pixor_data[i]
                        #pxr_det = pixor_data[i * 10 + 9]
                        pxr_clr = (255, 255, 0)
                        for det in pxr_det['objects']:
                            w = float(det['length'])
                            b = float(det['width'])  # width is in the y direction for Louis
                            x_c = float(det['centroid'][0])
                            y_c = float(det['centroid'][1])
                            theta = float(det['heading'])
                            pts = get_vertices(w, b, x_c, y_c, theta, img_h, img_w, scale / 100.)
                            img = draw_border(img, pts, pxr_clr, thiccness=5)

                        print (i)
                        if (i+1)%10 == 0 and i != 0:
                            labels = labels_data[((i+1)/10)-1]
                            l_clr = (255, 20, 147)
                            if labels['name'] != pxr_det['name']:
                                print "error mismatched label and pixor output pcd!"
                                print "pixor pcd: ", pxr_det['name']
                                print "label pcd: ", labels['name']
                                px_i += 1
                                continue

                            for label in labels['annotations']:
                                w = float(label['geometry']['dimensions']['x'])
                                b = float(label['geometry']['dimensions']['y'])
                                x_c = float(label['geometry']['position']['x'])
                                y_c = float(label['geometry']['position']['y'])
                                theta = float(label['geometry']['rotation']['z'])
                                pts = get_vertices(w, b, x_c, y_c, theta, img_h, img_w, scale / 100.)
                                if label['className'].find("Truck") >= 0 or label['className'].find("Bus") >= 0 or label[
                                    'className'].find("Car") >= 0 or label['className'].find("Van") >= 0:
                                    img = draw_border(img, pts, l_clr)
                                    cv2.putText(img, str(label['classId']), (int(round(img_w / 2.0 + (x_c * 100. / scale))),
                                                                             img_h - int(round(
                                                                                 img_h / 2.0 + y_c * 100. / scale)) + 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, l_clr, fthick, cv2.LINE_AA)
                                else:
                                    #                                 print "label class:" ,label['className']
                                    img = draw_border(img, pts, (100, 100, 100))
                                    cv2.putText(img, str(label['classId']), (int(round(img_w / 2.0 + (x_c * 100. / scale))),
                                                                             img_h - int(round(
                                                                                 img_h / 2.0 + y_c * 100. / scale)) + 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), fthick, cv2.LINE_AA)

                        # tracker 2 json --> green
                        frame = tracker_data2[i]
                        #frame = tracker_data2[i * 10 + 9]
                        for obj in frame['objects']:
                            track_clr = (50, 205, 50)
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
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, track_clr, fthick, cv2.LINE_AA)

                        # tracker json --> blue
                        frame = tracker_data[i]
                        #frame = tracker_data[i * 10 + 9]
                        for obj in frame['objects']:
                            track_clr = (0, 191, 255)
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
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, track_clr, fthick, cv2.LINE_AA)

                        # IBEO VALUES

                        frame = ibeo_data[i]
                        # frame = tracker_data2[i * 10 + 9]
                        for obj in frame['data']:
                            track_clr = (255, 165, 0)
                            if (len(obj) != 0):

                                obj_class = obj['obj_class']
                                width = float(obj['obj_size']['x']) / 100
                                length = float(obj['obj_size']['y']) / 100
                                # if obj_class == 6:
                                # print('Detected Truck!')
                                x_bus = float(obj['obj_center']['x']) / 100
                                y_bus = float(obj['obj_center']['y']) / 100



                                if  width > 2 and length > 1.5 and width < 10 and length < 5 and width != length  and np.abs( x_bus) < 35 and np.abs(y_bus) < 20 :
                                    ratiowl = round(width / length)

                                    if ratiowl > 1 and ratiowl < 5:

                                        #print('hi')
                                        w =  float(obj['obj_size']['x']) / 100
                                        b = float(obj['obj_size']['y']) / 100
                                        x_c = float(obj['obj_center']['x']) / 100
                                        y_c = float(obj['obj_center']['y']) / 100
                                        theta = ( float(obj['obj_orient_mdeg']))
                                        pts = get_vertices(w, b, x_c, y_c, theta, img_h, img_w, scale / 100.)
                                        img = draw_border(img, pts, track_clr)

                                        cv2.putText(img, str(obj_class), (int(round(img_w / 2.0 + (x_c * 100. / scale))),
                                                                          img_h - int(round(img_h / 2.0 + y_c * 100. / scale)) - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, track_clr, fthick, cv2.LINE_AA)
                                        class_certainty = (float(obj['class_certainty']))
                                        name =  "Width" + str(width) + " Length" + str(length) + " Class certainty" + str(class_certainty)

                                        x_p = int((x_c)*100/scale)
                                        y_p = int((y_c)*100/scale)
                                        cv2.putText(img, name, (int(round(img_w / 2.0 + (x_c * 100. / scale))),
                                                                          img_h - int(round(img_h / 2.0 + y_c * 100. / scale)) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_clr,  fthick)


                        # TODO must add to the bus pose
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.7
                        thickness = 1

                        cv2.putText(img, "Tracker_(Given to Linn)", (50, 50), font, fontScale,
                                    (50, 205, 50), thickness)
                        cv2.putText(img, "Tracker_(Improvements)", (50, 70), font, fontScale, (0, 191, 255), thickness)
                        cv2.putText(img, "PIXOR++", (50, 90), font, fontScale, (255, 255, 0), thickness)
                        cv2.putText(img, "Labels", (50, 110), font, fontScale, (255, 20, 147), thickness)
                        cv2.putText(img, "IBEO Values", (50, 130), font, fontScale, (255, 165, 0), thickness)
                        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        j = i+1
                        cv2.imwrite(img_path+"/img_"+str(j)+".jpg", im_rgb)
                        #cv2.imwrite(img_path + "/img_" + labels['name'][1:5] + ".jpg", im_rgb)

print "Done"

# In[ ]:





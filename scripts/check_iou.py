#!/usr/bin/env python
# coding: utf-8


# visualise data from json files
import json
import numpy as np
import cv2
import math
import pypcd
from datetime import datetime
from munkres import Munkres, print_matrix, DISALLOWED
import copy
from numba import jit
from shapely.geometry import box, Polygon

import sys
from __builtin__ import True


# check if it can be detected by pixorpp (if it is inside the px fov)
# returns True if ANY corner is in
# l_pts is list of vertices [[x,y], [x,y], ..]
def is_pxpp_fov(l_pts):
    px_fov_x = 35.2
    px_fov_y = 20.
    is_in = False
    for pt in l_pts:
        if abs(pt[0]) <= px_fov_x and abs(pt[1]) <= px_fov_y:
            is_in = True
            break
    #     if not is_in:
    #         print "label is outside of pixor fov: ", l_pts
    return is_in


def draw_border(img, pts, clr, thickness=2):
    for i in range(len(pts) - 1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), clr, thickness)
    cv2.line(img, tuple(pts[0]), tuple(pts[len(pts) - 1]), clr, thickness)
    return img


def get_vertices(w, b, x_c, y_c, theta, img_h, img_w, scale):
    pts = np.array([[]], dtype=float)

    ptx = x_c + w / 2.0 * math.cos(theta) - b / 2.0 * math.sin(theta)
    pty = y_c + w / 2.0 * math.sin(theta) + b / 2.0 * math.cos(theta)

    ptx = (1.0 * ptx / scale + img_w / 2.0)
    pty = (1.0 * pty / scale + img_h / 2.0)
    pts = np.append(pts, [ptx, pty])

    ptx = x_c - w / 2.0 * math.cos(theta) - b / 2.0 * math.sin(theta)
    pty = y_c - w / 2.0 * math.sin(theta) + b / 2.0 * math.cos(theta)

    ptx = (1.0 * ptx / scale + img_w / 2.0)
    pty = (1.0 * pty / scale + img_h / 2.0)
    pts = np.vstack((pts, [ptx, pty]))

    ptx = x_c - w / 2.0 * math.cos(theta) + b / 2.0 * math.sin(theta)
    pty = y_c - w / 2.0 * math.sin(theta) - b / 2.0 * math.cos(theta)

    ptx = (1.0 * ptx / scale + img_w / 2.0)
    pty = (1.0 * pty / scale + img_h / 2.0)
    pts = np.vstack((pts, [ptx, pty]))

    ptx = x_c + w / 2.0 * math.cos(theta) + b / 2.0 * math.sin(theta)
    pty = y_c + w / 2.0 * math.sin(theta) - b / 2.0 * math.cos(theta)

    ptx = (1.0 * ptx / scale + img_w / 2.0)
    pty = (1.0 * pty / scale + img_h / 2.0)
    pts = np.vstack((pts, [ptx, pty]))
    return pts


# In[6]:


@jit
def get_area(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# In[7]:


def get_iou(opts, tpts):
    if len(tpts) < 3 or len(opts) < 3:
        print "error in get_iou inputs: pts < 3"

    o_poly = Polygon(opts)
    t_poly = Polygon(tpts)
    print "Tracked polygon size",  get_area(tpts)
    print "Opts polygon size", get_area(opts)
    # The intersection
    area_intersect = o_poly.intersection(t_poly).area
    #     print "area of int: ", area_intersect
    area_union = get_area(opts) + get_area(tpts) - area_intersect
    iou = (100. * area_intersect) / area_union
    print "area_intersect = ", area_intersect
    print "area_union = ", area_union

    print "iou: ", iou

    return iou


# In[8]:


def get_MOT_dist(opts, tpts, dist_type):
    if dist_type == "IOU":
        IOU = get_iou(opts, tpts)
        dist = 100. - IOU
    else:
        "distance metric type not supported"
        dist = None
    return dist


# takes inputs in baselink frame
def is_in_PIXOR_FOV(label):
    px_fov_x = 35.199
    px_fov_y = 19.99

    w = float(label['geometry']['dimensions']['x'])
    b = float(label['geometry']['dimensions']['y'])
    x_c = float(label['geometry']['position']['x'])
    y_c = float(label['geometry']['position']['y'])
    theta = float(label['geometry']['rotation']['z'])

    return x_c >= -px_fov_x and x_c <= px_fov_x and y_c >= -px_fov_y and y_c <= px_fov_y


#   pts = get_vertices(w,b,x_c,y_c,theta, 0, 0, 1)
#   return is_pxpp_fov(pts)

def check_iou_json(labels_json_path, tracker_json_path, thres_d=100., distance_metric="IOU", is_write=True,
                   total_list=None):
    labels_json_file = open(labels_json_path)
    labels_data = json.load(labels_json_file, encoding="utf-8")

    if is_write:
        tracker_json_file = open(tracker_json_path)
        tracker_data = json.load(tracker_json_file, encoding="utf-8")
    else:
        tracker_data = total_list

    # initialise params
    total_dist = 0.
    num_matches = 0.
    mappings = {}  # label_id:tracker_id
    total_ct = 0
    total_gt = 0
    total_missed = 0
    total_mt = 0
    total_fpt = 0
    total_mmet = 0
    MOTP = None
    MOTA = None
    total_ct_check = 0
    l_index = 0  # offset of label and tracker in json
    t_index = 0
    #   missed_frames=0 # counting missed objects when tracker misses a frame. by right should never non-zero
    #   fp_frames=0

    for index, labels in enumerate(labels_data):  # for each pcd/timestep labelled
        # init params for each timestep
        mme_t = 0
        m_t = 0
        fp_t = 0
        new_mappings = {}

        # match label to tracker output ##############################################
        pcd_name = labels['name']
        time_step = pcd_name.split('.')[0]
        #               print "checking time step: ", time_step
        tracks = tracker_data[(index + 1 - l_index) * 10 - 1]  # FIXME this will only work if the files are 10 hz apart

        #print('Track name %s' % (tracks['name']))
        if tracks['name'] != pcd_name:
            print "Error: expected pcd file: ", pcd_name, "but instead is: ", tracks[
                'name'], "label and tracking json files do not match or has unconventional frequencies.\n", \
                "label n tracker data must be 10 hz apart"
            # raise ValueError # FIXME choose a more suitable error
            if pcd_name.split('.')[0] < tracks['name'].split('.')[0]:  # the tracker missed the label frame??
                while tracks['name'] != pcd_name:
                    l_index += 1
                    tracks = tracker_data[(index + 1 - l_index) * 10 - 1]
                    total_missed += len(labels['annotations'])
                continue
            else:
                # FIXME create empty 'label' to do mota with
                while tracks['name'] != pcd_name:
                    l_index -= 1
                    total_fpt += len(tracks["objects"])
                    # if((index+1-l_index)*10-1) < 1000:
                    print('Value to be checked %f ' % ((index + 1 - l_index) * 10 - 1))
                    if (index + 1 - l_index) * 10 - 1 < 1000:
                        tracks = tracker_data[(index + 1 - l_index) * 10 - 1]

        is_labelled = False
        corresp_label = None
        corresp_track = None

        for label_id, tracker_id in mappings.items():
            for obj_label in labels['annotations']:
                if is_in_PIXOR_FOV(obj_label):
                    if obj_label['classId'] == label_id:
                        is_labelled = True
                        corresp_label = obj_label
                        break

            is_tracked = False
            if is_labelled:
                for track in tracks['objects']:
                    if track['id'] == tracker_id:
                        is_tracked = True
                        corresp_track = track
                        break

            if is_labelled and is_tracked:
                if is_in_PIXOR_FOV(corresp_label):
                    ow = float(corresp_label['geometry']['dimensions']['x'])
                    ob = float(corresp_label['geometry']['dimensions']['y'])
                    ox_c = float(corresp_label['geometry']['position']['x'])
                    oy_c = float(corresp_label['geometry']['position']['y'])
                    otheta = float(corresp_label['geometry']['rotation']['z'])
                    #                     oobj_id = corresp_label['classId']
                    opts = get_vertices(ow, ob, ox_c, oy_c, otheta, 0, 0, 1)

                    tw = float(corresp_track['width'])
                    tb = float(corresp_track['length'])
                    tx_c = float(corresp_track['x'])
                    ty_c = float(corresp_track['y'])
                    ttheta = float(corresp_track['yaw'])
                    #                     tobj_id = int(corresp_track['id'])
                    tpts = get_vertices(tw, tb, tx_c, ty_c, ttheta, 0, 0, 1)

                    dist = get_MOT_dist(opts, tpts, distance_metric)

                    if dist < thres_d:
                        # corresponds
                        new_mappings.update({label_id: tracker_id})

        # get correspondance matrix:
        mat_size = max(len(labels['annotations']), len(tracks['objects']))
        corresp_mat = np.full((mat_size, mat_size), sys.maxint)
        #             corresp_mat = np.full((mat_size, mat_size), DISALLOWED)
        #             corresp_mat = np.zeros((mat_size, mat_size))
        for i, label in enumerate(labels['annotations']):
            for j, obj in enumerate(tracks['objects']):
                if is_in_PIXOR_FOV(label):
                    #                       print "checking label: ", label['classId']
                    ow = float(label['geometry']['dimensions']['x'])
                    ob = float(label['geometry']['dimensions']['y'])
                    ox_c = float(label['geometry']['position']['x'])
                    oy_c = float(label['geometry']['position']['y'])
                    otheta = float(label['geometry']['rotation']['z'])
                    #                     oobj_id = corresp_label['classId']
                    opts = get_vertices(ow, ob, ox_c, oy_c, otheta, 0, 0, 1)

                    tw = float(obj['width'])
                    tb = float(obj['length'])
                    tx_c = float(obj['x'])
                    ty_c = float(obj['y'])
                    ttheta = float(obj['yaw'])
                    #                     tobj_id = int(corresp_track['id'])
                    tpts = get_vertices(tw, tb, tx_c, ty_c, ttheta, 0, 0, 1)

                    corresp_mat[i, j] = get_MOT_dist(opts, tpts, distance_metric)

        #             print "corresp_mat: ", corresp_mat
        munkres_cost = copy.deepcopy(corresp_mat)
        #               print_matrix(munkres_cost, msg='correspondance mat:')

        # Munkres  algo
        m = Munkres()
        indexes = m.compute(munkres_cost)

        #             print indexes

        #             print "corresp_mat: ", corresp_mat

        #               print "munkres results:"
        for row, column in indexes:
            value = corresp_mat[row, column]
            #                   print row, column, value
            if value < thres_d:  # FIXME signage different for IOU and eucledian?
                #                     print "val: ", value, "; thresh: ", thres_d
                if labels['annotations'][row]['classId'] in mappings:
                    # TODO check for mismatched/contradictions
                    if mappings[labels['annotations'][row]['classId']] != tracks['objects'][column]['id']:
                        # change in corresponding tracking
                        mme_t += 1
                        # update mappings with new correspondance
                        new_mappings.update({labels['annotations'][row]['classId']: tracks['objects'][column]['id']})
                else:
                    # new correspondance
                    new_mappings.update({labels['annotations'][row]['classId']: tracks['objects'][column]['id']})

                total_dist += get_MOT_dist(opts, tpts, distance_metric)
                total_ct_check += 1

        total_ct += len(new_mappings)  # count number of matches

        mappings = copy.deepcopy(new_mappings)
        # match label to tracker output complete ######################################

        #               print "new mappings: ", new_mappings

        # calculate the false positives
        for obj in tracks['objects']:
            if obj['id'] not in mappings.values():
                # false positive
                fp_t += 1

        # calculate misses
        for label in labels['annotations']:
            if is_in_PIXOR_FOV(label):
                total_gt += 1
                if label['classId'] not in mappings:
                    # missed detection
                    m_t += 1

        #             total_gt += len(labels['annotations'])

        total_missed += m_t
        total_missed += fp_t
        total_missed += mme_t

        total_mt += m_t
        total_fpt += fp_t
        total_mmet += mme_t

    # MOTA = 1 - float(total_missed) / float(total_gt)
    # MOTP = float(total_dist) / float(total_ct)

    try:
        # calculate MOTP
        MOTP = total_dist / total_ct

        # calculate MOTA
        MOTA = 1. - (total_missed * 1.) / total_gt
    except:
        MOTA = -float('inf')
        MOTP = 0.
    #             print "error calculating MOTA"
    # #           print "MOTP: ", MOTP, "MOTA: ", MOTA
    #             print "total dist: ", total_dist, "\ntotal num of objects per frame:", total_gt
    #             print "total mt, fp, mme: ", total_mt, total_fpt, total_mmet
    #             print "total tracked: ", total_ct

    return MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt, total_missed


if __name__ == '__main__':
    # labels_json_path = "/media/yl/downloads/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/labels.old/Set_8_annotations.json"
    labels_json_path = "/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/set_1/labels/set1_annotations.json"

    distance_metric = "IOU"  # using IOU as distance metric
    thres_d = 100.  # 100 threshold distance to count as a correspondance, beyond it will be considered as missed detection
    # tracker_json_path = "/media/yl/downloads/tracker_results/set_8/newfp_cyra_statemax_age=8,min_hits=6,hung_thresh=0.25_Qqv_0.01953125.json"
    # tracker_json_path = "/media/yl/downloads/tracker_results/set_8/newfp_cyra_statemax_age=6,min_hits=3,hung_thresh=0.25_Qqv_10.0.json"
    today = datetime.today()
    d1 = today.strftime("%Y_%m_%d")
    set_num = '1'

    # tracker_json_path = "./results/JI_Cetran_Set1/TrackOutput_Set1_2020_02_10.json"
    tracker_json_path = "./results/JI_Cetran_Set1/TrackOutput_Set1_2020_03_04.json"

    # tracker_json_path = "./results/JI_Cetran_Set"+ set_num + "/TrackOutput_Set" + set_num + '_' + d1 + ".json"
    # tracker_json_path = "./results/JI_Cetran_Set1/yltracker_set_1.json"

    MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt, total_missed = check_iou_json(
        labels_json_path, tracker_json_path, thres_d, distance_metric)
    print MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt, total_missed


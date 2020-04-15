from __future__ import print_function
import os.path, copy, numpy as np, time, sys
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from utils import load_list_from_folder, fileparts, mkdir_if_missing
from scipy.spatial import ConvexHull
from pyquaternion import Quaternion
from wen_utils import STATE_SIZE, MEAS_SIZE, MOTION_MODEL, get_CV_F, get_CA_F, camRadarFuse, readCamera, readRadar,readJson, readLidar, readIBEO
import json
from datetime import datetime
from shapely.geometry import Polygon
import glob
from check_iou_jsons import check_iou_json
import subprocess

def happyTracker  (dataR , dataL , dataC , dataC_a3 , dataPose, dataIB,  max_age, min_hits, hung_thresh,
                               Rlidar, Qmodel, P_0lidar , Rcr, P_0cr, Ribeo, P_0ibeo, radarCam_threshold, radar_offset, testPIXOR, testIBEO, testCamDar):

    """""
    Function : happyTracker 
    Outputs : Trackers N x 8 array , where N is the number of tracked objects
     {"width": d[1], "height": d[0], "length": d[2], "x": T_track[0][0], "y": T_track[1][0], "z": d[5], "yaw": d[6], "id": d[7]}
    
    Inputs : dataR - Radar_obstacles, dataL (Pixor outputs) , dataC (Camera a0 outputs) , dataC_a3 (Camera a3 outputs), dataPose (Ego Pose)
    DEFAULT VALUES 
    max_age=3    
    min_hits=2
    hung_thresh=0.01 #.2
    
    LIDAR Measurement and Model values : Rlidar, Qlidar , P_0lidar
    Rlidar = np.identity(7)
    Qlidar = np.identity(14)  the covariance of the process noise;
    P_0lidar = np.identity(14) the covariance of the observation noise;
    
    Camera Radar Measurement and Model values : Rcr, Qcr , P_0cr
    Rcr = np.identity(7)
    Qcr = np.identity(14)
    P_0cr = np.identity(14)
    
    radarCam_threshold = 0.05  ; in radians the max angle difference +/- 0.025
    radar_offset = 0 ; the position of the radar point and the actual center of the ego vehicle 
    """""

    total_time = 0.0
    total_frames = 0
    seq_dets_pose =np.zeros([len(dataPose), 5])
    mot_tracker = AB3DMOT(is_jic=True, max_age=max_age, min_hits=min_hits, hung_thresh=hung_thresh,
                          Rlidar=Rlidar, Qmodel=Qmodel, P_0lidar=P_0lidar ,
                          Rcr=Rcr, P_0cr=P_0cr,Ribeo = Ribeo, P_0ibeo = P_0ibeo)
    #mot_tracker = AB3DMOT(is_jic, max_age, min_hits, hung_thresh, Rlidar, Qmodel, P_0lidar, Rcr, P_0cr, Ribeo, P_0ibeo)

    total_list = []

    #print ('Beginning tracking..')

    for frame_name in range(0, len(dataPose)):  # numPose
        #print(frame_name)
        #if frame_name == 162:
        if frame_name != 10000007:
            det_radar = dataR[frame_name]['front_esr_tracklist']
            det_radar_right = dataR[frame_name]['front_right_esr_tracklist']
            det_radar_left = dataR[frame_name]['front_left_esr_tracklist']
            det_radar_backL = dataR[frame_name]['rear_sbmp_tracklist']
            det_radar_backR = dataR[frame_name]['fsm4_tracklist']

            det_cam = dataC[frame_name]['objects']
            det_cam_back = dataC_a3[frame_name]['objects']
            det_lidar = dataL[frame_name]['objects']
            det_IBEO = dataIB[frame_name]['data']

            seq_dets_pose[frame_name][0] = frame_name
            seq_dets_pose[frame_name][1] = dataPose[frame_name]['header']['stamp']
            seq_dets_pose[frame_name][2] = dataPose[frame_name]['pose']['position']['x']
            seq_dets_pose[frame_name][3] = dataPose[frame_name]['pose']['position']['y']
            seq_dets_pose[frame_name][4] = dataPose[frame_name]['pose']['attitude']['yaw']
            q1 = Quaternion(axis=[0, 0, 1], angle=seq_dets_pose[frame_name][4])
            T1 = q1.transformation_matrix
            T1[0][3] = seq_dets_pose[frame_name][2]
            T1[1][3] = seq_dets_pose[frame_name][3]

            dets_radar = readRadar(frame_name, det_radar)
            dets_radar_right = readRadar(frame_name, det_radar_right)
            dets_radar_left = readRadar(frame_name, det_radar_left)
            dets_radar_backL = readRadar(frame_name, det_radar_backL)
            dets_radar_backR = readRadar(frame_name, det_radar_backR)

            dets_lidar, additional_info_lidar = readLidar(det_lidar, frame_name, T1)

            dets_cam = readCamera(frame_name, det_cam, camNum = 0)
            dets_radar_total_a0 = np.vstack((dets_radar, dets_radar_right, dets_radar_left))
            dets_camDar, additional_info_2 = camRadarFuse(frame_name, dets_cam, dets_radar_total_a0, T1, radarCam_threshold, 0 )

            dets_cam_back = readCamera(frame_name, det_cam_back, camNum=3)
            dets_radar_total_a3 = np.vstack((dets_radar_backL, dets_radar_backR))
            dets_camDar_back, additional_info_3 = camRadarFuse(frame_name, dets_cam_back, dets_radar_total_a3, T1 ,radarCam_threshold,3)


            dets_IBEO, additional_info_ibeo = readIBEO(frame_name, det_IBEO, T1)

            total_frames += 1

            start_time = time.time()
            #mot_tracker.frame_count += 1

            trackers = []

            if testPIXOR ==1 and (np.count_nonzero(dets_lidar) != 0):
                dets_all = {'dets': dets_lidar[:, 1:8], 'info': additional_info_lidar}
                trackers = mot_tracker.update(dets_all = dets_all, sensor_type = 1)

            if testCamDar == 1  and (np.count_nonzero(dets_camDar) != 0):
                dets_all2 = {'dets': dets_camDar[:, 1:8], 'info': additional_info_2}
                trackers = mot_tracker.update(dets_all =dets_all2 , sensor_type = 2)
            # #
            if testCamDar == 1 and (np.count_nonzero(dets_camDar_back) != 0):
                dets_all2 = {'dets': dets_camDar_back[:, 1:8], 'info': additional_info_3}
                trackers = mot_tracker.update(dets_all =dets_all2, sensor_type = 2)

            if testIBEO == 1 and(np.count_nonzero(dets_IBEO) != 0):
                dets_all2 = {'dets': dets_IBEO[:, 1:8], 'info': additional_info_ibeo}
                trackers = mot_tracker.update(dets_all =dets_all2, sensor_type = 3)

            if len(dets_lidar) == 0 and len(dets_camDar) == 0 and len(dets_camDar_back) == 0 and len(mot_tracker.trackers) >0 :                # dets_all = {'dets': empty_dets, 'info': empty_dets}
                # trackers = mot_tracker.update(dets_all=dets_all2, sensor_type=3)

                dets_all = {'dets':[],  'info': additional_info_lidar}
                trackers = mot_tracker.update(dets_all=dets_all, sensor_type=1)
                #print('No detections but still tracking!!')

                #trackers = mot_tracker.update(dets_all=dets_all2, sensor_type=3)

            #print('tracking..')
            cycle_time = time.time() - start_time
            total_time += cycle_time
            result_trks = []  # np.zeros([1,9])

            T_inv = np.linalg.inv(T1)
            T_tracked = np.zeros([4, 1])

            for d in trackers:

                T_tracked[0] = d[3]
                T_tracked[1] = d[4]
                T_tracked[3] = 1
                T_track = np.matmul(T_inv, T_tracked)

                #det_id2str = {0: 'Pedestrian', 1 : 'Bicycles', 2: 'PMD', 3: 'Motorbike', 4: 'Car' , 5: 'Truck', 6: 'Bus'}
                det_id2str = {0: 'Unknown', 1: 'Bicycles', 2: 'PMD', 3: 'Motorbike', 4: 'Car', 5: 'Truck', 6: 'Bus', 7: 'Pedestrian'}
                #type_tmp = det_id2str[d[9]]

                obj_dict = {"width": d[1], "height": d[0], "length": d[2], "x": T_track[0][0], "y": T_track[1][0],
                            "z": d[5], "yaw": d[6],
                            "id": d[7], "className": det_id2str[d[9]], "classType": d[9]}
                #
                # obj_dict = {"width": d[1], "height": d[0], "length": d[2], "x": T_track[0][0], "y": T_track[1][0],
                #            "z": d[5], "yaw": d[6],
                #            "id": d[9], "classNum": }

                result_trks.append(obj_dict)

            total_list.append({"name": dataL[frame_name]['name'], "objects": result_trks})

        #print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))


    return total_list


@jit  # let Numba decide when and how to optimize:
def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

@jit  #TODO understand where this corners came from
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c

@jit
def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)

    test = np.array(inter_p)

    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume

    else:
        return None, 0.0

def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        # if dc[0] * dp[1] - dc[1] * dp[0] == 0:
        #     return 0
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        #print('n3 values is : %d' % n3)
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)
#
# def iou3d(corners1, corners2):
#     ''' Compute 3D bounding box IoU.
#     Input:
#         corners1: numpy array (8,3), assume up direction is negative Y
#         corners2: numpy array (8,3), assume up direction is negative Y
#     Output:
#         iou: 3D bounding box IoU
#         iou_2d: bird's eye view 2D bounding box IoU
#
#     '''
#     # corner points are in counter clockwise order
#     rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
#     rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
#     area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
#     area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
#     inter, inter_area = convex_hull_intersection(rect1, rect2)
#     iou_2d = inter_area / (area1 + area2 - inter_area)
#     ymax = min(corners1[0, 1], corners2[0, 1])
#     ymin = max(corners1[4, 1], corners2[4, 1])
#     inter_vol = inter_area * max(0.0, ymax - ymin)
#     vol1 = box3d_vol(corners1)
#     vol2 = box3d_vol(corners2)
#     iou = inter_vol / (vol1 + vol2 - inter_vol)
#     return iou, iou_2d


def shapely_polygon_intersection(poly1, poly2):
    """
    """
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    return poly1.intersection(poly2).area


def test_shapely_polygon_intersection1():
    """
    """
    poly1 = np.array(
        [
            [0, 0],
            [3, 0],
            [3, 3],
            [0, 3]
        ])
    poly2 = np.array(
        [
            [2, 1],
            [5, 1],
            [5, 4],
            [2, 4]
        ])
    inter_area = shapely_polygon_intersection(poly1, poly2)
    assert inter_area == 2


def test_shapely_polygon_intersection2():
    """
    """
    poly1 = np.array(
        [
            [0, 0],
            [4, 0],
            [4, 4],
            [0, 4]
        ])
    poly2 = np.array(
        [
            [0, 0],
            [4, 0],
            [4, 4],
        ])
    inter_area = shapely_polygon_intersection(poly1, poly2)
    assert inter_area == 8


def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    inter_area = shapely_polygon_intersection(rect1, rect2)

    # inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d
#
# def iou2d(corners1, corners2):
#     ''' Compute 3D bounding box IoU.
#
#     Input:
#         corners1: numpy array (4,2), assume up direction is negative Y
#         corners2: numpy array (4,2), assume up direction is negative Y
#     Output:
#         iou_2d: bird's eye view 2D bounding box IoU
#     '''
#     # corner points are in counter clockwise order
#     rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
#     rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
#     area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
#     area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
#     inter, inter_area = convex_hull_intersection(rect1, rect2)
#     iou_2d = inter_area / (area1 + area2 - inter_area)
#     # ymax = min(corners1[0,1], corners2[0,1])
#     # ymin = max(corners1[4,1], corners2[4,1])
#     # inter_vol = inter_area * max(0.0, ymax-ymin)
#     # vol1 = box3d_vol(corners1)
#     # vol2 = box3d_vol(corners2)
#     # iou = inter_vol / (vol1 + vol2 - inter_vol)
#     return iou_2d

#FIXED Not using the silly convex hull thing..

def iou2d(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    inter_area = shapely_polygon_intersection(rect1, rect2)
    # inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    # ymax = min(corners1[0, 1], corners2[0, 1])
    # ymin = max(corners1[4, 1], corners2[4, 1])
    # inter_vol = inter_area * max(0.0, ymax - ymin)
    # vol1 = box3d_vol(corners1)
    # vol2 = box3d_vol(corners2)
    # iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou_2d

@jit
def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

@jit
def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def convert_3dbox_to_8corner(bbox3d_input):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    R = roty(bbox3d[3])

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]

    return np.transpose(corners_3d)

def associate_detections_to_trackers(detections, trackers, iou_threshold):
    # def associate_detections_to_trackers(detections,trackers,iou_threshold=0.01):     # ablation study
    # def associate_detections_to_trackers(detections,trackers,iou_threshold=0.25):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    detections:  N x 8 x 3
    trackers:    M x 8 x 3

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            #       iou_matrix[d,t] = iou3d(det,trk)[0]             # det: 8 x 3, trk: 8 x 3
            iou_matrix[d, t] = iou2d(det, trk)  # det: 8 x 3, trk: 8 x 3
    matched_indices = linear_assignment(-iou_matrix)  # hugarian algorithm

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class AB3DMOT(object):

    def __init__(self, is_jic, max_age, min_hits, hung_thresh, Rlidar, Qmodel, P_0lidar, Rcr, P_0cr, Ribeo, P_0ibeo):

        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.is_jic = is_jic
        self.hungarian_thresh = hung_thresh

        self.Q = Qmodel

        self.R = Rlidar
        self.P_0 = P_0lidar

        self.Rcr = Rcr
        self.P_0cr = P_0cr

        self.Rlidar = Rlidar
        self.P_0lidar = P_0lidar

        self.Ribeo = Ribeo
        self.P_0ibeo = P_0ibeo

        self.delta_t = 0.05  # FIXME make it variable for fusion/ live usage

    def update(self, dets_all, sensor_type):
        # type: (object, object) -> object
        """
        Params:
          dets_all: dict
            dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
            info: a array of other info for each det
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        dets, info = dets_all['dets'], dets_all['info']  # dets: N x 7, float numpy array

        if sensor_type == 1:  # LIDAR
            self.P_0  = self.P_0lidar
            self.R = self.Rlidar

        if sensor_type == 2:  # camera and radar
            self.P_0  = self.P_0cr
            self.R = self.Rcr
            # FIXME does camera radar need it's own MA & MH?
            # self.max_age = max_age
            # self.min_hits = min_hits

        if sensor_type == 3:  # ibeo
            self.P_0  = self.P_0ibeo
            self.R = self.Ribeo

        if not self.is_jic:
            dets = dets[:,self.reorder]  # in the /data files the order is: h w l x y z R (which needs to be reordered to be x y z theta l w h


        #FIXME Sensor fusion framecount??
        self.frame_count += 1

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 7))  # N x 7 ,
        to_del = []
        ret = []
        for t, trk in enumerate(trks):  # t=index trk=0
            pos = self.trackers[t].predict().reshape((-1, 1))  # predicted state of t-th tracked item
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]  # predicted state of t-th tracked item
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # ????
        for t in reversed(to_del):  # delete tracked item if cannot predict state?! #FIXME
            self.trackers.pop(t)

        # does NOT project anything, just gives corners in 3D space
        #if len(dets) != 0 :

        dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]

        if len(dets_8corner) > 0:
            dets_8corner = np.stack(dets_8corner, axis=0)
        else:
            dets_8corner = []
        trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
        if len(trks_8corner) > 0:
            trks_8corner = np.stack(trks_8corner, axis=0)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner,
                                                                                   self.hungarian_thresh)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                trk.update(dets[d, :][0], info[d, :][0])

        # create and initialise new trackers for unmatched detections BUT we shouldn't for camera ?????
        #if sensor_type != 2: Cannot birth ONLY with lidar and ibeo ........

        for i in unmatched_dets:  # a scalar of index
            trk = KalmanBoxTracker(dets[i, :], info[i, :], self.R, self.Q, self.P_0, self.delta_t)
            self.trackers.append(trk)

        if sensor_type == 1:  # LIDAR
            self.P_0lidar = self.P_0
            self.Rlidar = self.R

        if sensor_type == 2:  # camera and radar
            self.P_0cr = self.P_0
            self.Rcr = self.R

        if sensor_type == 3:  # ibeo
            self.P_0ibeo = self.P_0
            self.Ribeo = self.R


        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()  # bbox location
            d = d[self.reorder_back]

            # choose which tracks to return

            #FIXME
            if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
            #if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits)):
                ret.append(
                    np.concatenate((d, [trk.id + 1], trk.info)).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)  # x, y, z, theta, l, w, h, ID, other info, confidence

        return np.empty((0, 15))

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox3D, info, R, Q, P_0, delta_t):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=STATE_SIZE, dim_z=MEAS_SIZE)
        if MOTION_MODEL == "CV":
            self.kf.F = get_CV_F(delta_t)
        elif MOTION_MODEL == "CA":
            self.kf.F = get_CA_F(delta_t)
        elif MOTION_MODEL == "CYRA":
            self.kf.F = get_CYRA_F(delta_t)
        else:
            print("unknown motion model", MOTION_MODEL)
            raise ValueError

        # x y z theta l w h
        self.kf.H = np.zeros((MEAS_SIZE, STATE_SIZE))
        for i in range(min(MEAS_SIZE, STATE_SIZE)):
            self.kf.H[i, i] = 1.

        self.kf.R[0:, 0:] = R  # measurement uncertainty

        # initialisation cov
        #     self.kf.P[7:,7:] *= 1000. #state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        #     self.kf.P *= 10.

        self.kf.P = P_0
        # self.kf.P *= 10.
        # self.kf.P[7:, 7:] *= 1000.  #state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        # self.kf.Q[-1,-1] *= 0.01
        # self.kf.Q[7:,7:] *= 0.01 # process uncertainty
        self.kf.Q = Q
        self.kf.x[:7] = bbox3D.reshape((7, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 1  # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.info = info  # other info

        self.historytrack = info[0]

    def update(self, bbox3D, info):
        """
        Updates the state vector with observed bbox.
        """


        self.time_since_update = 0


        self.history = []

        #if info[0] != self.historytrack:
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the fist time

        #
        # else:
        #     self.historytrack = info[0]

        ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(
                new_theta - predicted_theta) < np.pi * 3 / 2.0:  # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        #########################

        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        self.info = info

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7,))

if __name__ == '__main__':

    print("Initialising...")

    # det_id2str = {0: 'Pedestrian', 2: 'Car', 3: 'Cyclist', 4: 'Motorcycle' , 5: 'Truck'}
    # det_id2str = {0: 'Pedestrian', 1 : 'Bicycles', '2: 'PMD', 3: 'Motorbike', 4: 'Car' , 5: 'Truck', 6: 'Bus'}

    basedir = '/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/set_1/'

    testCamDar = 1
    testPIXOR = 1
    testIBEO = 1

    pathRadar = basedir + '/radar_obstacles/radar_obstacles.json'
    pathCamera_a0 = basedir + '/image_detect/result_a0.json'
    pathCamera_a1 = basedir + '/image_detect/result_a1.json'
    pathCamera_a2 = basedir + '/image_detect/result_a2.json'
    pathCamera_a3 = basedir + '/image_detect/result_a3.json'

    pathLidar = basedir + '/pixor_outputs_pixorpp_kitti_nuscene.json'
    pathIBEO = basedir + '/ecu_obj_list/ecu_obj_list.json'
    pathPose = basedir + '/fused_pose/fused_pose.json'

    #
    # #
    # basedir_total = ['/media/wen/demo_ssd/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/16_sep/log_high/set_8',
    #                  '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_3',
    #                  '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_2',
    #                  '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_1',
    #                  '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_12',
    #                  '/media/wen/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_3',
    #                  '/media/wen/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_9']
    # labels_total = ['/media/wen/demo_ssd/raw_data/train_labels/JI_ST-cloudy-day_2019-08-27-21-55-47/set_8',
    #                 '/media/wen/demo_ssd/raw_data/train_labels/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/set_3',
    #                 '/media/wen/demo_ssd/raw_data/train_labels/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/set_2',
    #                 '/media/wen/demo_ssd/raw_data/train_labels/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/set_1',
    #                 '/media/wen/demo_ssd/raw_data/train_labels/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/set_12',
    #                 '/media/wen/demo_ssd/raw_data/train_labels/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/set_3',
    #                 '/media/wen/demo_ssd/raw_data/train_labels/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/set_9']


    # #
    basedir_total = ['/media/wen/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_2']
    labels_total = ['/media/wen/demo_ssd/raw_data/eval_labels/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/set_2/']

    print('Trying testcase 12566!! ')
    i = 0 #to be the one with pedesterians
    basedir = basedir_total[i]
    print(basedir)
    labels_json_path = glob.glob(labels_total[i] + "/*annotations.json")
    print(labels_json_path)
    # Join various path components
    pathRadar = os.path.join(basedir, "radar_obstacles/radar_obstacles.json")
    pathCamera_a0 = glob.glob(basedir + "/image_detections/results_cama0*.json")[0]
    pathCamera_a3 = glob.glob(basedir + "/image_detections/results_cama3*.json")[0]
    pathLidar = basedir + '/pixor_outputs_mdl_tf_epoch_150_valloss_0.2106.json'
    print (pathLidar)
    pathIBEO = basedir + '/ecu_obj_list/ecu_obj_list.json'
    pathPose = basedir + '/fused_pose/fused_pose.json'
    #

    rng_thres = np.array([0.01, 0.1, 1, 10, 100])


    dataR , dataL , dataC , dataC_a3 , dataPose, dataIB = readJson(pathRadar, pathLidar, pathCamera_a0, pathCamera_a3, pathPose, pathIBEO)


    max_age=6
    min_hits= 3
    rlA = 0
    rlB = 0
    rlC = 2
    rlD = 0
    rlE = 0

    hung_thresh = 0.01  # hung_thresh_total[ht]
    Rlidar = np.identity(7)
    Rlidar[2, 2] = 10. ** -5  # z
    Rlidar[6, 6] = 10. ** -5  # h

    Qmodel = np.identity(14)
    # tuning
    Qmodel[0][0] *= rng_thres[rlA]
    Qmodel[1][1] = Qmodel[0][0]
    Qmodel[3][3] *= rng_thres[rlB]

    P_0lidar = np.identity(14)
    # tuning
    P_0lidar[0][0] *= rng_thres[rlC]
    P_0lidar[1][1] = P_0lidar[0][0]

    Rcr = np.identity(7)
    Rcr[0, 0] = 0.001  # error in x and y !! for camera radar fusion
    Rcr[1, 1] = 0.001  # error in x and y !! for camera radar fusion
    Rcr[2, 2] = 10. ** -5  # z
    Rcr[6, 6] = 10. ** -5  # h

    P_0cr = np.identity(14)

    # tuning
    P_0cr[0][0] *= rng_thres[rlD]
    P_0cr[1][1] = P_0cr[0][0]


    Ribeo = np.identity(7)
    Ribeo[0, 0] = 0.01  # 10cm 0.1*0.1   0.01
    Ribeo[1, 1] = 0.01  # 10cm 0.1*0.1   0.01
    Ribeo[2, 2] = 10. ** -5  # z
    Ribeo[6, 6] = 10. ** -5  # h

    P_0ibeo = np.identity(14)
    # # tuning
    P_0ibeo[0][0] *= rng_thres[rlE]
    P_0ibeo[1][1] = P_0ibeo[0][0]


    radarCam_threshold = 0.1  # .05 #radians!!
    radar_offset = 0


    total_list = happyTracker (dataR , dataL , dataC , dataC_a3 , dataPose, dataIB,  max_age, min_hits, hung_thresh,
                               Rlidar, Qmodel, P_0lidar , Rcr, P_0cr, Ribeo, P_0ibeo, radarCam_threshold, radar_offset, testPIXOR, testIBEO, testCamDar)

    isPrint = 1
    isCheckIOU = 1
    isVisualise = 0

    if isPrint == True:
        today = datetime.today()
        d1 = today.strftime("%Y_%m_%d")
        tracker_json_outfile = "/home/wen/AB3DMOT/scripts/results/sensorfusion/checkSF.json"
        with open(tracker_json_outfile, "w+") as outfile:
            json.dump(total_list, outfile, indent=1)

        print('Saved tracking results as Json')

    if isCheckIOU == True:
        labels_json_path =  glob.glob(labels_total[i] + "/*annotations.json")
        print (labels_json_path)
        #labels_json_path = '/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/set_1/labels/set1_annotations.json'
        distance_metric = "IOU"  # using IOU as distance metric
        thres_d = 100.  # 100 threshold distance to count as a correspondance, beyond it will be considered as missed detection

        MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = check_iou_json(labels_json_path[0],
                                                                                                     tracker_json_outfile,
                                                                                                     thres_d,
                                                                                                     distance_metric)
        print(MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt)


    if isVisualise == True:
        subprocess.call(['python', 'dataset_visualisation.py'])


#!/usr/bin/env python
# coding: utf-8

'''
 file for utils for CHRISTINAAAAAAAAA
'''

import json
import numpy as np
import cv2
import os

STATE_SIZE = 14
MEAS_SIZE = 7   #measurement model for pixor, 7
#MEAS_SIZE_Radar = 3  #measurement model for radar, 3
#MOTION_MODEL = "CYRA"
#MOTION_MODEL="CA"
MOTION_MODEL="CV"

# set R based on pixor stats in json
def px_stats_get_R(pixor_stats_json):
    with open(pixor_stats_json) as pixor_stats_file:
        data = json.load(pixor_stats_file, encoding="utf-8")
        var = data['var']
        # x y z theta l w h
        R = np.identity(MEAS_SIZE)  # KF measurement uncertainty/noise
        # tp, fp, fn, [x y w b theta]
        # pixor_outputs_tf_epoch_3_valloss_0.0093.json
        # 210 92 335 [0.03953874 0.00588307 0.02431999 0.39831919 0.00211127] precision@80%iou75 :  69.54%, recall:  54.69%
        R[0, 0] = var[0]  # x
        R[1, 1] = var[1]  # y
        R[2, 2] = 10. ** -5  # z
        R[3, 3] = var[4]  # theta
        R[4, 4] = var[3]  # l
        R[5, 5] = var[2]  # w
        R[6, 6] = 10. ** -5  # h
    return R


def px_stats_get_P_0(pixor_stats_json, p0_v=1000., factor=1.):
    with open(pixor_stats_json) as pixor_stats_file:
        data = json.load(pixor_stats_file, encoding="utf-8")
        var = data['var']
        # x y z theta l w h
        P_0 = np.identity(STATE_SIZE)  # KF measurement uncertainty/noise
        # tp, fp, fn, [x y w b theta]
        # pixor_outputs_tf_epoch_3_valloss_0.0093.json
        # 210 92 335 [0.03953874 0.00588307 0.02431999 0.39831919 0.00211127] precision@80%iou75 :  69.54%, recall:  54.69%
        P_0[0, 0] = var[0]  # x
        P_0[1, 1] = var[1]  # y
        P_0[2, 2] = 0.  # z
        P_0[3, 3] = var[4]  # theta
        P_0[4, 4] = var[3]  # l
        P_0[5, 5] = var[2]  # w
        P_0[6, 6] = 0.  # h
        P_0[6, 6] = 0.  # h
        P_0 = P_0 * factor
        P_0[7, 7] = p0_v  # vx
        P_0[8, 8] = p0_v  # vy
        P_0[9, 9] = 0.  # vz
        # print "P_O", P_0
    return P_0


def get_CV_Q(q_v, delta_t):
    Q = np.zeros((STATE_SIZE, STATE_SIZE))
    Q[0, 0] = delta_t ** 3 * q_v / 3.
    Q[1, 1] = delta_t ** 3 * q_v / 3.
    Q[0, 7] = delta_t ** 2 * q_v / 2.
    Q[1, 8] = delta_t ** 2 * q_v / 2.
    Q[7, 0] = delta_t ** 2 * q_v / 2.
    Q[8, 1] = delta_t ** 2 * q_v / 2.
    Q[7, 7] = delta_t * q_v
    Q[8, 8] = delta_t * q_v
    # print "Q", Q
    return Q


def get_CV_F(delta_t):
    F = np.eye(STATE_SIZE)
    F[0, 7] = delta_t
    F[1, 8] = delta_t
    F[2, 9] = delta_t
    return F


def get_CA_Q(q_a, delta_t):
    Q = np.zeros((STATE_SIZE, STATE_SIZE))
    Q[0, 0] = delta_t ** 5 * q_a / 20.
    Q[1, 1] = delta_t ** 5 * q_a / 20.
    Q[0, 7] = delta_t ** 4 * q_a / 8.
    Q[1, 8] = delta_t ** 4 * q_a / 8.
    Q[0, 10] = delta_t ** 3 * q_a / 6.
    Q[1, 11] = delta_t ** 3 * q_a / 6.
    Q[7, 0] = delta_t ** 4 * q_a / 8.
    Q[8, 1] = delta_t ** 4 * q_a / 8.
    Q[10, 0] = delta_t ** 3 * q_a / 6.
    Q[11, 1] = delta_t ** 3 * q_a / 6.
    Q[7, 7] = delta_t ** 3 * q_a / 3.
    Q[8, 8] = delta_t ** 3 * q_a / 3.
    Q[7, 10] = delta_t ** 2 * q_a / 2.
    Q[8, 11] = delta_t ** 2 * q_a / 2.
    Q[7, 10] = delta_t ** 2 * q_a / 2.
    Q[8, 11] = delta_t ** 2 * q_a / 2.
    Q[10, 10] = delta_t * q_a
    Q[11, 11] = delta_t * q_a
    return Q


def get_CA_F(delta_t):
    F = np.eye(STATE_SIZE)
    F[0, 7] = delta_t
    F[1, 8] = delta_t
    F[0, 10] = delta_t ** 2. / 2.
    F[1, 11] = delta_t ** 2. / 2.
    F[7, 10] = delta_t
    F[8, 11] = delta_t
    return F


def get_CYRA_Q(q_a, q_p, T):
    Q = np.zeros((STATE_SIZE, STATE_SIZE))
    Q[0, 0] = T ** 5 * q_a / 20.
    Q[1, 1] = T ** 5 * q_a / 20.
    Q[2, 2] = T ** 2 * q_p / 3.
    Q[0, 7] = T ** 4 * q_a / 8.
    Q[1, 8] = T ** 4 * q_a / 8.
    Q[0, 10] = T ** 2 * q_a / 6.
    Q[1, 11] = T ** 2 * q_a / 6.
    Q[2, 13] = T ** 2 * q_p / 2.
    Q[7, 0] = T ** 4 * q_a / 8.
    Q[8, 1] = T ** 4 * q_a / 8.
    Q[7, 7] = T ** 2 * q_a / 3.
    Q[8, 8] = T ** 2 * q_a / 3.
    Q[7, 10] = T ** 2 * q_a / 2.
    Q[8, 11] = T ** 2 * q_a / 2.
    Q[10, 0] = T ** 2 * q_a / 6.
    Q[11, 1] = T ** 2 * q_a / 6.
    Q[13, 2] = T ** 2 * q_p / 2.
    Q[10, 7] = T ** 2 * q_a / 2.
    Q[11, 8] = T ** 2 * q_a / 2.
    Q[10, 10] = T * q_a
    Q[11, 11] = T * q_a
    Q[13, 13] = T * q_p
    return Q


def get_CYRA_F(delta_t):
    F = np.eye(STATE_SIZE)
    F[0, 7] = delta_t
    F[1, 8] = delta_t
    F[0, 10] = delta_t ** 2. / 2.
    F[1, 11] = delta_t ** 2. / 2.
    F[2, 13] = delta_t
    F[7, 10] = delta_t
    F[8, 11] = delta_t
    return F

def HJradar(x):    #For radar measurement H

    dist = np.sqrt(x[0][0]**2 + x[1][0]**2)
    d = np.zeros((3, 14), dtype=float)

    d[0][0] = x[0][0] / dist
    d[0][1] = x[1][0] / dist

    dist2 = dist**3

    d[1][0] = (- x[1][0] * (x[1][0] * x[8][0] - x[0][0] * x[9][0])) / dist2
    d[1][1] = (- x[0][0] * (x[0][0] * x[9][0] - x[1][0] * x[8][0])) / dist2

    d[1][8] = x[0][0] / dist
    d[1][9] = x[1][0] / dist

    d[2][0] = - x[0][0] / (dist**2)
    d[2][1] = x[1][0] / (dist**2)

    return d

def hxRadar(x):

    temp = 2
    range = np.sqrt(x[0][0]**2 + x[1][0]**2)
    rangerate = (x[0][0]*x[8][0] + x[1][0]*x[9][0]) / range

    if x[0][0] > 0:
        temp = x[1][0] / x[0][0]

    theta = np.arctan(temp)

    return array ([[range, rangerate, theta]]).reshape((dim_z, 1))

def camRadarFuse(frame_name, dets_cam, dets_radar, T1, radarCam_threshold):
    dets_camDar = np.zeros([1, 9])
    additional_info_2 = np.zeros([1, 7])
    numCamDar = 0

    for w in range(len(dets_cam)):
        test_x = dets_cam[w][1]
        #radarCam_threshold = 0.05
        bestTrack_Radar = -1
        for q in range(len(dets_radar)):
            test_radar = dets_radar[q][3]
            diff = test_radar - test_x
            if (abs(diff) < radarCam_threshold):
                radarCam_threshold = diff
                bestTrack_Radar = q

        if (dets_cam[w][2] == 4):  #4 ==car
            length = 5  # 5
            width = 2.5  # 2.5
        if (dets_cam[w][2] == 5):  # 5 == truck
            length = 8 # 5
            width = 3  # 2.5
        if (dets_cam[w][2] == 6):  # 6 == bus!!
            length = 12 # 5
            width = 3  # 2.5
        if (dets_cam[w][2] <= 3):  #0 == pedesterians/bicycles/pmd/motorbike!!
            length = 1  # 5
            width = 1  # 2.5
        # if (dets_cam[w][2] == 0):  #0 == pedesterians!!
        #     print('Human detected on camera radar !!!!')

        sidebounds = 20 #25 #20
        frontbackbounds = 35.2 #40 #35.2

        if (bestTrack_Radar != -1 and abs(dets_radar[bestTrack_Radar][1]) < frontbackbounds and abs(
                dets_radar[bestTrack_Radar][2] < sidebounds)):
            pos_cr = np.zeros([4, 1])
            # if (dets_cam[w][2] == 0):  #0 == pedesterians!!
            #     print('Human detected on camera radar !!!!')
            #print('radar similar point: %s' % (q))
            k = numCamDar
            dets_camDar[k][0] = frame_name
            dets_camDar[k][1] = dets_radar[bestTrack_Radar][1]
            dets_camDar[k][2] = dets_radar[bestTrack_Radar][2]
            dets_camDar[k][3] = 1
            dets_camDar[k][4] = 0 #-dets_radar[bestTrack_Radar][4] #dets_radar[q][3]
            dets_camDar[k][5] = width
            dets_camDar[k][6] = length
            dets_camDar[k][7] = 1  # HEIGHT
            dets_camDar[k][8] = 2  # sensor type: 2

            pos_cr[0][0] = dets_camDar[k][1]
            pos_cr[1][0] = dets_camDar[k][2]
            pos_cr[2][0] = 0
            pos_cr[3][0] = 1

            T_cr = np.matmul(T1, pos_cr)

            dets_camDar[k][1] = T_cr[0][0]
            dets_camDar[k][2] = T_cr[1][0]
            additional_info_2[k, 1] = dets_cam[w][2]

    return dets_camDar, additional_info_2

def undistort_points(xy_tuple, K, D):
    pts = np.array([int(x) for x in xy_tuple])
    Knew = K.copy()
    #print(np.array([[pts]]).shape)

    upts = [int(x) for x in cv2.fisheye.undistortPoints(np.array([[[float(x) for x in pts]]]),K=K, D=D, P=Knew)[0][0]]
    return upts

def readCamera(frame_name, det_cam):
    h = 0
    #CAMERA INTRINSICS
    Camera_Matrix_GMSL_120 = np.array([[958.5928517660333, 0.0, 963.2848327985546], [0.0, 961.1122866843237, 644.5199995337151], [0.0, 0.0, 1.0]])  #
    ftest = 0.5 *( Camera_Matrix_GMSL_120[0][0] + Camera_Matrix_GMSL_120[1][1])
    test = 963.2848327985546 * float(416)/float(1920)
    f = ftest / test  # focal length fx
    f_new = Camera_Matrix_GMSL_120[0][0]
    #print(det_cam)
    sf = 0.5
    dets_cam = np.zeros([len(det_cam), 5])
    K = np.array([[981.276*sf, 0.0, 985.405*sf],
                  [0.0, 981.414*sf, 629.584*sf],
                  [0.0, 0.0, 1.0]])
    D = np.array([[-0.0387077],[-0.0105319],[-0.0168433],[0.0310624]])

    for j in range(len(det_cam)):
        dets_cam[h][0] = frame_name
        cam_x = det_cam[j]['relative_coordinates']['center_x'] * 960
        cam_y = det_cam[j]['relative_coordinates']['center_y'] * 604
        xy_tuple = (cam_x, cam_y)
        upts = undistort_points(xy_tuple, K, D)
        c2 = -upts[0]  + (960 / 2)
        f3 = Camera_Matrix_GMSL_120[0][0]  #* float(416)/float(1920)
        theta = np.arctan(float(c2)/ f3) #FIXME Verify if the theta is correct
        dets_cam[h][1] = theta

        # det_id2str = {0: 'Pedestrian', 2: 'Car', 3: 'Cyclist', 4: 'Motorcycle' , 5: 'Truck'}
        type = det_cam[j]['class_id']  # class_id = 2 is a car

        #print type
        if type == 0:
            type_sf = 0

        if type == 1:
            type_sf = 1
        if type == 2:
            type_sf = 4
        if type == 3:
            type_sf = 3
        if type == 4:
            type_sf = 6
        if type == 5:
            type_sf = 5

        #det_id2str = {0: 'Pedestrian', 1: 'Bicycles', 2: 'PMD', 3: 'Motorbike', 4: 'Car', 5: 'Truck', 6: 'Bus'}

        dets_cam[h][2] = type_sf
        dets_cam[h][3] = det_cam[j]['confidence']
        dets_cam[h][4] = 3  # SENSOR TYPE = 3
        h += 1
    return dets_cam

def undistort_unproject_pts(xy_tuple):
    """
    This function converts existing values into the undistorted values
    """
    # sfx = float(416)/float(1920)  # scaling factor
    # sfy = float(416)/float(1208)
    # K = np.array([[981.276 * sfx, 0.0, 985.405 * sfx],
    #               [0.0, 981.414 * sfy, 629.584 * sfy],
    #               [0.0, 0.0, 1.0]])
    # D = np.array([[-0.0387077], [-0.0105319], [-0.0168433], [0.0310624]])
    # pts = np.array([int(x) for x in xy_tuple])
    # #print(pts)
    # Knew = K.copy()
    # upts = [int(x) for x in cv2.fisheye.undistortPoints(np.array([[[float(x) for x in pts]]]),K=K, D=D, P=Knew)[0][0]]
    # #print(upts)
    # #x = int((512. + (point_out[0][0][0] * 1024)))
    # #y = int((288. + (point_out[0][0][1] * 576)))
    # return upts

    sf = 0.5 # scaling factor
    K = np.array([[981.276*sf, 0.0, 985.405*sf],
                  [0.0, 981.414*sf, 629.584*sf],
                  [0.0, 0.0, 1.0]])
    D = np.array([[-0.0387077],[-0.0105319],[-0.0168433],[0.0310624]])

    pts = np.array([int(x) for x in xy_tuple])
    Knew = K.copy()
    #print(np.array([[pts]]).shape)
    upts = [int(x) for x in cv2.fisheye.undistortPoints(np.array([[[float(x) for x in pts]]]),K=K, D=D, P=Knew)[0][0]]
    return upts


def readRadar(frame_name, det_radar, radar_offset):
    i = 0
    dets_radar = np.zeros([len(det_radar), 6])
    for j in range(len(det_radar)):
        dets_radar[i][0] = frame_name
        dets_radar[i][1] = det_radar[j]['obj_vcs_posex']
        dets_radar[i][2] = det_radar[j]['obj_vcs_posey']
        rangerate = float(det_radar[j]['range_rate'])
        #print(rangerate)
        # dets_radar[i][1] = dets_radar[i][1] + 0.1*rangerate
        # dets_radar[i][2] = dets_radar[i][2] + 0.1*rangerate

        # ## To compensate of the fact that the radar value doesn't give the centroid information....
        # if rangerate < 0:
        #     dets_radar[i][1] = dets_radar[i][1] - radar_offset
        #     dets_radar[i][2] = dets_radar[i][2] - radar_offset
        # else:
        #     dets_radar[i][1] = dets_radar[i][1] + radar_offset
        #     dets_radar[i][2] = dets_radar[i][2] + radar_offset

        dets_radar[i][3] = np.arctan(dets_radar[i][2] / dets_radar[i][1])
        angle = float(det_radar[j]['angle_centroid'])
        dets_radar[i][4] = np.deg2rad(angle)
        dets_radar[i][5] = 1  # sensor type: 1
        i += 1

    return dets_radar

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

#def readJson(pathRadar, pathLidar, pathCamera_a0, pathCamera_a1, pathCamera_a2, pathCamera_a3, pathPose, pathIBEO):
def readJson(pathRadar, pathLidar, pathCamera_a0, pathCamera_a3, pathPose, pathIBEO):

    # # Read the set1 radar points
    # # Read the set1 radar points
    with open(pathRadar, "r") as json_file:
        dataR = json.load(json_file).get('radar')

    # # Read the set1 lidar points
    with open(pathLidar, "r") as json_file:
        dataL = json.load(json_file)

    # Read the a0 camera points
    with open(pathCamera_a0, "r") as json_file:
        dataC = json.load(json_file)

#     # Read the a0 camera points
#     with open(pathCamera_a1, "r") as json_file:
#         dataC_a1 = json.load(json_file)
#
# # Read the set1 camera points
#     with open(pathCamera_a2, "r") as json_file:
#         dataC_a2 = json.load(json_file)

    # Read the set1 camera points
    with open(pathCamera_a3, "r") as json_file:
        dataC_a3 = json.load(json_file)

    # Read the ego pose
    with open(pathPose, "r") as json_file:
        pose = json.load(json_file)
        dataPose = pose.get('ego_loc')

    with open(pathIBEO, "r") as json_file:
        dataIB = json.load(json_file) .get('ibeo_obj')

#    return dataR, dataL, dataC, dataC_a1, dataC_a2, dataC_a3, dataPose, dataIB
    return dataR, dataL, dataC, dataC_a3, dataPose, dataIB

def readLidar (det_lidar, frame_name, T1):

    dets_lidar = np.zeros([len(det_lidar), 9])
    pos_lidar = np.zeros([4, 1])
    k = 0
    for j in range(len(det_lidar)):
        dets_lidar[k][0] = frame_name
        dets_lidar[k][1] = (det_lidar[j]['centroid'])[0]  # x values in pixor , but y in world frame!!
        dets_lidar[k][2] = (det_lidar[j]['centroid'])[1]  # y values in pixor , but x in world frame
        dets_lidar[k][3] = 1
        dets_lidar[k][4] = det_lidar[j]['heading']
        dets_lidar[k][5] = det_lidar[j]['width']  # width is in the y direction for Louis
        dets_lidar[k][6] = det_lidar[j]['length']
        dets_lidar[k][7] = 1
        dets_lidar[k][8] = 2  # sensor type: 2

        pos_lidar[0][0] = dets_lidar[k][1]
        pos_lidar[1][0] = dets_lidar[k][2]
        pos_lidar[2][0] = 0
        pos_lidar[3][0] = 1
        T2 = np.matmul(T1, pos_lidar)
        dets_lidar[k][1] = T2[0][0]
        dets_lidar[k][2] = T2[1][0]

        k += 1
        pos_lidar = np.zeros([4, 1])

    return dets_lidar

def readIBEO(frame_name, det_IBEO, T1):
    dets_IBEO_temp = np.zeros([1, 9])
    pos_IBEO = np.zeros([4, 1])
    k = 0
    additional_info_2_temp = np.zeros([1, 7])
    dets_IBEO = []
    additional_info_2 = []

    for j in range(len(det_IBEO)):
        obj_class = det_IBEO[j]['obj_class']
        width = float(det_IBEO[j]['obj_size']['x'])/100
        length = float(det_IBEO[j]['obj_size']['y'])/100
        #if obj_class == 6:
            #print('Detected Truck!')
        x_bus = float(det_IBEO[j]['obj_center']['x']) / 100
        y_bus = float(det_IBEO[j]['obj_center']['y']) / 100

        if obj_class > 3 and obj_class != 7 and obj_class < 10 and width != 0 and length != 0 and np.abs(x_bus) < 35 and np.abs(y_bus) < 20 :

            if obj_class == 3:
                obj_class_folobus = 0
                print("Pedesterian detected from IBEO!!")
            if obj_class == 9 :
                obj_class_folobus = 1
            if obj_class == 5:
                obj_class_folobus = 4
            if obj_class == 8 or obj_class == 4:
                obj_class_folobus = 3
                # if obj_class == 5:
                #     obj_class_foloyolo = 4
            if obj_class == 6: #TRUCKO
                obj_class_folobus = 5
                if width < 5:
                    width = 5

            dets_IBEO_temp[0][0] = frame_name
            dets_IBEO_temp[0][1] = x_bus # x values in pixor , but y in world frame!!
            dets_IBEO_temp[0][2] = y_bus  # y values in pixor , but x in world frame
            dets_IBEO_temp[0][3] = 1
            dets_IBEO_temp[0][4] = det_IBEO[j]['yaw']
            dets_IBEO_temp[0][5] = length  # width is in the y direction for Louis
            dets_IBEO_temp[0][6] = width
            dets_IBEO_temp[0][7] = 1
            dets_IBEO_temp[0][8] = 4  # sensor type: 4 IBEO!!!

            pos_IBEO[0][0] = dets_IBEO_temp[0][1]
            pos_IBEO[1][0] = dets_IBEO_temp[0][2]
            pos_IBEO[2][0] = 0
            pos_IBEO[3][0] = 1
            T2 = np.matmul(T1, pos_IBEO)
            dets_IBEO_temp[0][1] = T2[0][0]
            dets_IBEO_temp[0][2] = T2[1][0]
            additional_info_2_temp[0, 1] = obj_class_folobus
            if k == 0:
                dets_IBEO = np.copy(dets_IBEO_temp)
                additional_info_2 = np.copy(additional_info_2_temp)

            else:
                dets_IBEO = np.vstack((dets_IBEO, dets_IBEO_temp))
                additional_info_2 = np.vstack((additional_info_2, additional_info_2_temp))
            k += 1
            # print ('k is = %d' %k)
            pos_IBEO = np.zeros([4, 1])

 #   print (dets_IBEO, additional_info_2)

    return dets_IBEO, additional_info_2
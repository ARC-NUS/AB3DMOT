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
from wen_utils import STATE_SIZE, MEAS_SIZE, MEAS_SIZE_Radar, MOTION_MODEL, get_CV_F, get_CA_F
import json
from datetime import datetime
#from main import AB3DMOT

def happyTracker (dataR , dataL , dataC , dataC_a3 , dataPose, max_age, min_hits, hung_thresh, R, Q, P_0, Rcr, Qcr, P_0cr, radarCam_threshold, radar_offset):

    """""
    Function : happyTracker 
    Inputs : dataR - Radar_obstacles, dataL (Pixor outputs) , dataC (Camera a0 outputs) , dataC_a3 (Camera a3 outputs), dataPose (Ego Pose) 
    Outputs : Trackers N x 8 array , where N is the number of tracked objects
     {"width": d[1], "height": d[0], "length": d[2], "x": T_track[0][0], "y": T_track[1][0], "z": d[5], "yaw": d[6], "id": d[7]}
    
    DEFAULT VALUES 
    max_age=3
    min_hits=2
    hung_thresh=0.01 #.2
    R = np.identity(7)
    Q = np.identity(14)
    P_0 = np.identity(14)
    
    radarCam_threshold = 0.05  ; in radians the max angle difference +/- 0.025
    radar_offset = 0.7 ; the position of the radar point and the actual center of the ego vehicle 
    """""

    total_time = 0.0
    total_frames = 0
    seq_dets_pose =np.zeros([len(dataPose), 5])
    delta_t = 0.05
    mot_tracker = AB3DMOT(max_age=max_age, min_hits=min_hits, hung_thresh=hung_thresh, R=R, Q=Q, P_0=P_0, Rcr = Rcr, Qcr = Qcr, P_0cr = P_0cr, delta_t= delta_t, is_jic = False)
    total_list = []

    for frame_name in range(0, len(dataPose)):  # numPose
        k = 0
        det_radar = dataR[frame_name]['front_esr_tracklist']
        det_radar_right = dataR[frame_name]['front_right_esr_tracklist']
        det_radar_left = dataR[frame_name]['front_left_esr_tracklist']
        det_radar_back = dataR[frame_name]['rear_sbmp_tracklist']
        det_cam = dataC[frame_name]['objects']
        det_cam_back = dataC_a3[frame_name]['objects']

        det_lidar = dataL[frame_name]['objects']
        seq_dets_pose[frame_name][0] = frame_name
        seq_dets_pose[frame_name][1] = dataPose[frame_name]['header']['stamp']
        seq_dets_pose[frame_name][2] = dataPose[frame_name]['pose']['position']['x']
        seq_dets_pose[frame_name][3] = dataPose[frame_name]['pose']['position']['y']
        seq_dets_pose[frame_name][4] = dataPose[frame_name]['pose']['attitude']['yaw']

        q1 = Quaternion(axis=[0, 0, 1], angle=seq_dets_pose[frame_name][4])
        T1 = q1.transformation_matrix
        T1[0][3] = seq_dets_pose[frame_name][2]
        T1[1][3] = seq_dets_pose[frame_name][3]

        print("Processing ", dataL[frame_name]['name'],
              datetime.utcfromtimestamp(seq_dets_pose[frame_name][1]).strftime('%Y-%m-%d %H:%M:%S'))

        dets_radar = readRadar(frame_name, det_radar, radar_offset)
        dets_radar_right = readRadar(frame_name, det_radar_right, radar_offset)
        dets_radar_left = readRadar(frame_name, det_radar_left, radar_offset)
        dets_radar_back = readRadar(frame_name, det_radar_back, radar_offset)

        dets_radar = np.concatenate((dets_radar, dets_radar_right, dets_radar_left), axis=0)
        dets_lidar = np.zeros([len(det_lidar), 9])

        pos_lidar = np.zeros([4, 1])
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

        seq_dets_total = dets_lidar
        additional_info = np.zeros([len(seq_dets_total), 7])
        additional_info[:, 1] = 2

        dets_cam = readCamera(frame_name, det_cam)
        dets_cam_back = readCamera(frame_name, det_cam_back)
        dets_camDar, additional_info_2 = camRadarFuse(frame_name, dets_cam, dets_radar, T1, radarCam_threshold)
        dets_camDar_back, additional_info_3 = camRadarFuse(frame_name, dets_cam_back, dets_radar_back, T1 ,radarCam_threshold)

        total_frames += 1

        dets_all = {'dets': seq_dets_total[:, 1:8], 'info': additional_info}
        start_time = time.time()
        trackers = []
        trackers = mot_tracker.update(dets_all = dets_all, sensor_type = 1)

        if (np.count_nonzero(dets_camDar) != 0):
            seq_dets_total2 = dets_camDar
            additional_info2 = additional_info_2
            dets_all2 = {'dets': seq_dets_total2[:, 1:8], 'info': additional_info2}
            trackers = mot_tracker.update(dets_all =dets_all2 , sensor_type = 2)
        # #
        if (np.count_nonzero(dets_camDar_back) != 0):
            seq_dets_total2 = dets_camDar_back
            additional_info2 = additional_info_3
            dets_all2 = {'dets': seq_dets_total2[:, 1:8], 'info': additional_info2}
            trackers = mot_tracker.update(dets_all =dets_all2 , sensor_type = 2)

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

            obj_dict = {"width": d[1], "height": d[0], "length": d[2], "x": T_track[0][0], "y": T_track[1][0],
                        "z": d[5], "yaw": d[6],
                        "id": d[7]}

            result_trks.append(obj_dict)
        total_list.append({"name": dataL[frame_name]['name'], "objects": result_trks})

    return total_list

def readJson (pathRadar , pathLidar , pathCamera_a0 , pathCamera_a3 , pathPose):

    # # Read the set1 radar points
    with open(pathRadar, "r") as json_file:
        dataR = json.load(json_file).get('radar')

    # # Read the set1 lidar points
    with open(pathLidar, "r") as json_file:
        dataL = json.load(json_file)

    # Read the set1 a0 camera points
    with open(pathCamera_a0, "r") as json_file:
        dataC = json.load(json_file)

    # Read the set1 camera points
    with open(pathCamera_a3, "r") as json_file:
        dataC_a3 = json.load(json_file)

    # Read the ego pose
    with open(pathPose, "r") as json_file:
        pose = json.load(json_file)
        dataPose = pose.get('ego_loc')

    return dataR , dataL , dataC , dataC_a3 , dataPose

def readLidar (det_lidar, frame_name):
    pos_lidar = np.zeros([4, 1])
    k = 0
    dets_lidar = np.zeros([1, 9])
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

def undistort_unproject_pts(xy_tuple):
    """
    This function converts existing values into the undistorted values
    """
    sf = 0.5  # scaling factor
    K = np.array([[981.276 * sf, 0.0, 985.405 * sf],
                  [0.0, 981.414 * sf, 629.584 * sf],
                  [0.0, 0.0, 1.0]])
    D = np.array([[-0.0387077], [-0.0105319], [-0.0168433], [0.0310624]])

    pts = np.array([int(x) for x in xy_tuple])
    Knew = K.copy()
    print(np.array([[pts]]).shape)
    ux, uy = [int(x) for x in cv2.fisheye.undistortPoints(np.array([[[float(x) for x in pts]]]),K=K, D=D, P=Knew)[0][0]]

    return ux, uy

def camRadarFuse(frame_name, dets_cam, dets_radar, T1, radarCam_threshold):
    dets_camDar = np.zeros([1, 9])
    additional_info_2 = np.zeros([1, 7])
    numCamDar = 0

    for w in range(len(dets_cam)):
        test_x = dets_cam[w][1]
        radarCam_threshold = 0.05
        bestTrack_Radar = -1
        for q in range(len(dets_radar)):
            test_radar = dets_radar[q][3]
            diff = test_radar - test_x
            if (abs(diff) < radarCam_threshold):
                radarCam_threshold = diff
                bestTrack_Radar = q

        if (dets_cam[w][2] == 2):
            length = 5  # 5
            width = 2.5  # 2.5
        else:
            length = 8
            width = 3

        sidebounds = 20 #25 #20
        frontbackbounds = 35.2 #40 #35.2

        if (bestTrack_Radar != -1 and abs(dets_radar[bestTrack_Radar][1]) < frontbackbounds and abs(
                dets_radar[bestTrack_Radar][2] < sidebounds)):
            pos_cr = np.zeros([4, 1])
            print('radar similar point: %s' % (q))
            k = numCamDar
            dets_camDar[k][0] = frame_name
            dets_camDar[k][1] = dets_radar[bestTrack_Radar][1]
            print(dets_camDar[k][1])
            dets_camDar[k][2] = dets_radar[bestTrack_Radar][2]
            print(dets_camDar[k][2])
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

def readCamera(frame_name, det_cam):
    h = 0

    #CAMERA INTRINSICS
    Camera_Matrix_GMSL_120 = np.array([[958.5928517660333, 0.0, 963.2848327985546], [0.0, 961.1122866843237, 644.5199995337151], [0.0, 0.0, 1.0]])  #
    ftest = 0.5 *( Camera_Matrix_GMSL_120[0][0] + Camera_Matrix_GMSL_120[1][1])
    f = ftest / 1000  # focal length fx

    dets_cam = np.zeros([len(det_cam), 5])
    for j in range(len(det_cam)):
        dets_cam[h][0] = frame_name
        cam_x = -det_cam[j]['relative_coordinates']['center_x'] + 0.5
        theta = np.arctan(cam_x / f)

        #DONE : Change the distortion??
        # theta = theta + 0.005 * np.sin(abs(theta))

        dets_cam[h][1] = theta
        type = det_cam[j]['class_id']  # class_id = 2 is a car
        dets_cam[h][2] = type
        dets_cam[h][3] = det_cam[j]['confidence']
        dets_cam[h][4] = 3  # SENSOR TYPE = 3
        h += 1
    return dets_cam

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

        ## To compensate of the fact that the radar value doesn't give the centroid information....
        if rangerate < 0:
            dets_radar[i][1] = dets_radar[i][1] - radar_offset
            dets_radar[i][2] = dets_radar[i][2] - radar_offset
        else:
            dets_radar[i][1] = dets_radar[i][1] + radar_offset
            dets_radar[i][2] = dets_radar[i][2] + radar_offset

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
        if np.any(np.isnan(test)):
            print('gg')
            return None, 0.0
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
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


# FIXME change from 3d to 2d IOU checker, check if correct or not
def iou2d(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (4,2), assume up direction is negative Y
        corners2: numpy array (4,2), assume up direction is negative Y
    Output:
        iou_2d: bird's eye view 2D bounding box IoU
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    # ymax = min(corners1[0,1], corners2[0,1])
    # ymin = max(corners1[4,1], corners2[4,1])
    # inter_vol = inter_area * max(0.0, ymax-ymin)
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
    #def __init__(self, max_age=2, min_hits=3, hung_thresh=0.1, is_jic=False,
          #        R=np.identity(7), Q=np.identity(10), P_0=np.identity(10),
          #        delta_t=0.05):  # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
    def __init__(self, max_age, min_hits, hung_thresh, R, Q, P_0, Rcr, Qcr, P_0cr, delta_t, is_jic = False):
        # def __init__(self,max_age=3,min_hits=3):        # ablation study
        # def __init__(self,max_age=1,min_hits=3):
        # def __init__(self,max_age=2,min_hits=1):
        # def __init__(self,max_age=2,min_hits=5):
        # """
        # """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.is_jic = is_jic
        self.hungarian_thresh = hung_thresh

        self.R = R
        self.Q = Q
        self.P_0 = P_0

        self.Rcr = Rcr
        self.Qcr = Qcr
        self.P_0cr = P_0cr


        self.delta_t = delta_t  # FIXME make it variable for fusion/ live usage

    def update(self, dets_all, sensor_type):
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

        if not self.is_jic:
            dets = dets[:,
                   self.reorder]  # in the /data files the order is: h w l x y z R (which needs to be reordered to be x y z theta l w h
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
        dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
        if len(dets_8corner) > 0:
            dets_8corner = np.stack(dets_8corner, axis=0)
        else:
            dets_8corner = []
        trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
        if len(trks_8corner) > 0: trks_8corner = np.stack(trks_8corner, axis=0)

        # data association(?)
        #     matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers_BEV(dets, trks)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner,
                                                                                   self.hungarian_thresh)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                trk.update(dets[d, :][0], info[d, :][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            if sensor_type == 1: #LIDAR
                trk = KalmanBoxTracker(dets[i, :], info[i, :], self.R, self.Q, self.P_0, self.delta_t)

            if sensor_type == 2: #camera and radar
                trk = KalmanBoxTracker(dets[i, :], info[i, :], self.Rcr, self.Qcr, self.P_0cr, self.delta_t)
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()  # bbox location
            d = d[self.reorder_back]

            # choose which tracks to return
            if ((trk.time_since_update < self.max_age) and (
                    trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
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
    This class represents the internel state of individual tracked objects observed as bbox.
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

        # innov cov from pixor stats
        self.kf.P = P_0

        # self.kf.Q[-1,-1] *= 0.01
        #     self.kf.Q[7:,7:] *= 0.01 # process uncertainty
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

    def update(self, bbox3D, info):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the fist time

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

    det_id2str = {0: 'Pedestrian', 2: 'Car', 3: 'Cyclist', 4: 'Motorcycle' , 5: 'Truck'}

    # #################################
    ##  INSERT PATHS!!!

    pathRadar = '/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/set_1/radar_obstacles/radar_obstacles.json'
    pathLidar = '/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/set_1/pixor_outputs2_pixorpp_kitti_nuscene.json'
    pathCamera_a0 = '/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/set_1/image_detect/result_a0.json'
    pathCamera_a3 = '/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/set_1/image_detect/result_a3.json'
    pathPose = '/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/set_1/fused_pose/fused_pose.json'

    dataR , dataL , dataC , dataC_a3 , dataPose = readJson(pathRadar, pathLidar, pathCamera_a0, pathCamera_a3, pathPose)

    max_age=3
    min_hits=2
    hung_thresh=0.05 #.2
    R = np.identity(7)
    Q = np.identity(14)
    P_0 = np.identity(14)

    Rcr = np.identity(7)
    Qcr = np.identity(14)
    P_0cr = np.identity(14)

    radarCam_threshold = 0.01 #.05 #radians!!
    radar_offset = 0.7
    total_list = happyTracker (dataR , dataL , dataC , dataC_a3 , dataPose, max_age, min_hits, hung_thresh, R, Q, P_0, Rcr, Qcr, P_0cr, radarCam_threshold, radar_offset)



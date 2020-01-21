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
from wen_utils import STATE_SIZE, MEAS_SIZE, MEAS_SIZE_Radar, MOTION_MODEL, get_CV_F
import json
from datetime import datetime


def HJacobian(x):  # 3 by 14 matrix

    dist = np.sqrt(x[0][0] ** 2 + x[1][0] ** 2)
    d = np.zeros((3, 14), dtype=float)

    d[0][0] = x[0][0] / dist
    d[0][1] = x[1][0] / dist

    dist2 = dist ** 3

    d[1][0] = (- x[1][0] * (x[1][0] * x[8][0] - x[0][0] * x[9][0])) / dist2
    d[1][1] = (- x[0][0] * (x[0][0] * x[9][0] - x[1][0] * x[8][0])) / dist2

    d[1][8] = x[0][0] / dist
    d[1][9] = x[1][0] / dist

    d[2][0] = - x[0][0] / (dist ** 2)
    d[2][1] = x[1][0] / (dist ** 2)

    return d  # range rangerate theta


def hx(x):  # 3 by 1 matrix
    d = []
    temp = 2
    range = np.sqrt(x[0][0] ** 2 + x[1][0] ** 2)
    rangerate = (x[0][0] * x[8][0] + x[1][0] * x[9][0]) / range

    if x[0][0] > 0:
        temp = x[1][0] / x[0][0]

    theta = np.arctan(temp)
    # d = np.arange(3).reshape((dim_z, 1))
    d = np.array([range, rangerate, theta]).reshape((3, 1))
    return d

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
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        #print('%d' % n3)
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


@jit
def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


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

class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox3D, info):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=STATE_SIZE, dim_z=MEAS_SIZE)
        self.kfr = ExtendedKalmanFilter(dim_x=STATE_SIZE, dim_z=MEAS_SIZE_Radar)

        if MOTION_MODEL == "CV":
            #self.kf.F = get_CV_F(delta_t)
            self.kf.F = get_CV_F(0.05)
        else:
            print("unknown motion model", MOTION_MODEL)

        """
        Initilise for diff sensors 
        """
    # x y z theta l w h
        self.kf.H = np.zeros((MEAS_SIZE,STATE_SIZE))
        for i in range(min(MEAS_SIZE,STATE_SIZE)):
          self.kf.H[i,i]=1.

        self.kf.R[0:,0:] *= 10   # measurement uncertainty
        self.kf.P[7:,7:] *= 1000  # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.

        #self.kf.Q[7:, 7:] = Q

        self.kf.Q[7:, 7:] *= 0.01   # process uncertainty
        self.kf.x[:7] = bbox3D.reshape((7, 1))

        self.kfr.H = np.zeros((MEAS_SIZE_Radar, STATE_SIZE))

        for i in range(min(MEAS_SIZE_Radar, STATE_SIZE)):
          self.kfr.H[i, i] = 1.

        #self.kfr.x[:7] = bbox3D.reshape((7, 1))

        self.kf.R[0:,0:] *= 100.  #R  # measurement uncertainty
        self.kf.P[7:,7:] *= 1000.  # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *=10 # P_0  #10.

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

    def updateRadar(self, z):
        #rk.update(z, HJacobian, hx)
        #z = np.array([35.041, 0.1, 1]) # range, range rate, angle centroid THESE VALUES MUST BE UPDATED WITH THE READING ONES!!

        z = z.reshape(3, 1)
        self.kfr.x = self.kf.x
        HJ = HJacobian(self.kfr.x)
        #HJ = HJ.reshape(3,14)

        hxr = hx(self.kfr.x)
        hxr = hxr.reshape(3,1)
        
        self.kfr.update(z, HJacobian, hx)
        self.kfr.predict
        #print(self.kfr.x)

        # self.kfr.xs.append(rk.x)

        return self.kfr.x

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.01):      #self.hungarian_thresh
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
            iou_matrix[d, t] = iou3d(det, trk)[0]  # det: 8 x 3, trk: 8 x 3
    matched_indices = linear_assignment(-iou_matrix)  # hungarian algorithm

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
   def __init__(self,max_age=2, min_hits=2, is_jic=False,
               R = np.identity(14), Q = np.identity(14), P_0=np.identity(14), delta_t=0.05):      # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
  # def __init__(self,max_age=3,min_hits=3):        # ablation study
  # def __init__(self,max_age=1,min_hits=3):
  # def __init__(self,max_age=2,min_hits=1):
  # def __init__(self,max_age=2,min_hits=5):
    """
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.reorder = [3, 4, 5, 6, 2, 1, 0]
    self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
    self.is_jic = is_jic
    self.hungarian_thresh = 0.1 # hung_thresh

    self.R = R
    self.Q = 0 #Q
    self.P_0 = P_0
    self.delta_t = delta_t #TODO Use for fusion

   def update(self, dets_all):
        """
        Params:
          dets_all: dict
            dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
            info: a array of other info for each det
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        dets_lidar, dets_radar,  info, seq_dets_pose, dets_cam = dets_all['dets_lidar'], dets_all ['dets_radar'], dets_all['info'], dets_all ['seq_dets_pose'], dets_all['dets_cam']  # dets: N x 7, float numpy array
        dets = dets_lidar[:, self.reorder]

        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 7))  # N x 7 , #get predicted locations from existing trackers.
        trkR = np.zeros((len(dets_radar), 7))  # N x 7 , #get predicted locations from existing trackers.
        bestTrack= np.zeros((1,7))
        to_del = []
        ret = []

        # for i in range(len(dets_cam)):
        #     test_x = dets_cam[i][1]
        #     min = 0.1
        #     for j in range(len(dets_radar)):
        #         test_radar = dets_radar[j][2]
        #         diff = test_radar-test_x
        #         if(abs(diff)< min):
        #             min = diff
        #             bestTrack_Radar = j
        #     print ('radar similar point: %s' %(j))


        # #TODO to create and initialise new trackers for new radar detections
        #
        # for i in range(len(self.trackers)):
        #     #diffx = self.trackers[i].kf.x[5] - seq_dets_pose[self.frame_count][2]
        #     #diffy = self.trackers[i].kf.x[6] - seq_dets_pose[self.frame_count][3]
        #     track_theta = self.trackers[i].kf.x[0]
        #     min = 0.1
        #     for j in range(len(dets_radar)):
        #         dets_r = dets_radar[j]
        #         theta_ref = dets_radar[j][2]
        #         diff_theta = abs(theta_ref-track_theta)
        #         if (diff_theta < min and diff_theta > 0  ):
        #             min = diff_theta
        #             pos2 =self.trackers[i].updateRadar(dets_r)
        #             #print(pos2[0][0])
        #             bestTrack[0] = np.array([pos2[0][0], pos2[1][0], pos2[2][0], pos2[3][0], pos2[4][0], pos2[5][0], pos2[6][0]])

            #dets = np.concatenate(dets,bestTrack)


        for t, trk in enumerate(trks):
            #self.trackers
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]


            if (np.any(np.isnan(pos))):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
        if len(dets_8corner) > 0:
            dets_8corner = np.stack(dets_8corner, axis=0)
        else:
            dets_8corner = []
        trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
        if len(trks_8corner) > 0: trks_8corner = np.stack(trks_8corner, axis=0)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                trk.update(dets[d, :][0], info[d, :][0])             #UPDATE Values for lidar!!

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            #trk = KalmanBoxTracker(dets[i, :], info[i, :])
            trk = KalmanBoxTracker(dets[i, :], info[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()  # bbox location
            d = d[self.reorder_back]

            # print('trk.time_since_update = %d   max_age = %d' % (trk.time_since_update, self.max_age))
            # print('trk.hits = %d   self.min_hits = %d' % (trk.hits, self.min_hits))
            if ((trk.time_since_update < self.max_age) and (
                    trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(
                    np.concatenate((d, [trk.id + 1], trk.info)).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(i)
                # print('pop')

        if (len(ret) > 0):
            return np.concatenate(ret)  # x, y, z, theta, l, w, h, ID, other info, confidence
        return np.empty((0, 15))


if __name__ == '__main__':

    print("Initialise the testing of AB3DMOT")

    # if len(sys.argv) != 2:
    #     print("Usage: python main.py result_sha(e.g., car_3d_det_test)")
    #     #sys.exit(1)

    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    result_sha_2 = "JI_Cetran_Set1"
    save_root = './results'
    total_time = 0.0
    total_frames = 0
    save_dir = os.path.join(save_root, result_sha_2)

    mkdir_if_missing(save_dir)
    eval_dir = os.path.join(save_dir, 'data');
    mkdir_if_missing(eval_dir)


    # #################################
    # # Take in bag value names
    mainJson_loc = '../../raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_low/set_1/'

    #Read the set1 radar points
    pathJson = mainJson_loc + '/radar_obstacles/radar_obstacles.json'
    with open(pathJson, "r") as json_file:
        dataR = json.load(json_file)
        dataR = dataR.get('radar')

    #Read the set1 lidar points
    pathJson = mainJson_loc + '/pixor_outputs_tf_epoch_49_valloss_0.0117.json' #/pixor_outputs.json'
    with open(pathJson, "r") as json_file:
        dataL = json.load(json_file)


    #Read the set1 camera points
    pathJson = mainJson_loc + '/image_detect/result.json'
    with open(pathJson, "r") as json_file:
        dataC = json.load(json_file)

    #Read the ego pose
    pathJson = mainJson_loc + '/fused_pose/fused_pose_new.json'
    with open(pathJson, "r") as json_file:
        pose = json.load(json_file)
        dataPose = pose.get('ego_loc')
        numPose = len(dataPose)

    #READ GROUND TRUTH LABELS
    pathJson = mainJson_loc + '/labels/set1_annotations.json'
    with open(pathJson, "r") as json_file:
        dataLabels = json.load(json_file)
        GT = 0
        GT_indiv = np.zeros(len(dataLabels))

    for i in range(len(dataLabels)):
        det = dataLabels[i].get('annotations')
        GT_indiv[i] = len(det)
        GT += len(det)

    # example of detection for radar : frame , range, range_rate, theta , sensor_type =1
    seq_dets_radar = np.zeros([1, 5])

    # example of detection for lidar : frame ,, x, y, z, theta, l, w, h, sensor_type = 2
    seq_dets_lidar = np.zeros([1, 9])

    #TODO : Camera detection
    # example of detection for camera : frame , x, y
    seq_dets_cam = np.zeros([1, 4])
    seq_dets_pose =np.zeros([numPose, 5])

    mot_tracker = AB3DMOT()

    eval_file = os.path.join(eval_dir, 'Set_1_maxage %d minhits %d .txt' %(mot_tracker.max_age, mot_tracker.min_hits));
    eval_file = open(eval_file, 'w')

    Camera_Matrix_GMSL_120 = np.array([[958.5928517660333, 0.0, 963.2848327985546], [0.0, 961.1122866843237, 644.5199995337151], [0.0, 0.0, 1.0]])  #
    fx = Camera_Matrix_GMSL_120[0][0] / 1000 #focal length fx #TODO set the units!!

    for frame_name in range(1, numPose): #numPose
        i = 0
        k = 0
        h = 0
        #frame_name = frame_name+1
        det_radar = dataR[frame_name].get('front_esr_tracklist')
        det_cam = dataC[frame_name].get('objects')
        #det_lidar = dataL[ "%05d" % frame_name + '.pcd']
        det_lidar =dataL[frame_name].get('objects')
        seq_dets_pose[frame_name][0] = frame_name
        seq_dets_pose[frame_name][1] = dataPose[frame_name].get('header').get('stamp')
        seq_dets_pose[frame_name][2] = dataPose[frame_name].get('pose').get('position').get('x')
        seq_dets_pose[frame_name][3] = dataPose[frame_name].get('pose').get('position').get('y')
        seq_dets_pose[frame_name][4] = dataPose[frame_name].get('pose').get('attitude').get('yaw')

        q1 = Quaternion(axis=[0, 0, 1], angle=seq_dets_pose[frame_name][4])
        T1 = q1.transformation_matrix
        T1[0][3] = seq_dets_pose[frame_name][2]
        T1[1][3] = seq_dets_pose[frame_name][3]

        print("Processing %04d." % frame_name, datetime.utcfromtimestamp(seq_dets_pose[frame_name][1]).strftime('%Y-%m-%d %H:%M:%S'))
        dets_radar = np.zeros([len(det_radar), 5])

        ##Load Detections!
        for j in range(len(det_radar)):
            dets_radar[i][0] = frame_name
            dets_radar[i][1] = det_radar[j].get('range')
            dets_radar[i][2] = det_radar[j].get('range_rate')  #TODO how to convert this to required frame??
            dets_radar[i][3] = float(det_radar[j].get('angle_centroid')) * (np.pi / 180)
            dets_radar[i][4] = 1  #sensor type: 1

            #FRONT ESR Radar
            Bus_radar = np.array([[8.69], [0], [1.171]])

            q1 = Quaternion(axis=[1, 0, 0], angle=0.00349066)
            q2 = Quaternion(axis=[0, 1, 0], angle=-0.00872665)
            q3 = Quaternion(axis=[0, 0, 1], angle=0)

            q_radar = q1* q2* q3

            #todo to transform the points into x and y
            i += 1

        if frame_name == 1:
            seq_dets_radar = dets_radar
        else:
            seq_dets_radar = np.concatenate((seq_dets_radar, dets_radar), axis=0)


        dets_lidar = np.zeros([len(det_lidar), 9])

        #Load Lidar Detections
        for j in range(len(det_lidar)):
            dets_lidar[k][0] = frame_name
            dets_lidar[k][1] = (det_lidar[j].get('centroid'))[0]  # x values   #TODO use lidar quarternion to transform it :o
            dets_lidar[k][2] = (det_lidar[j].get('centroid'))[1]  # y values
            dets_lidar[k][3] = 1
            dets_lidar[k][4] = det_lidar[j].get('heading')
            dets_lidar[k][5] = det_lidar[j].get('length')
            dets_lidar[k][6] = det_lidar[j].get('width')
            dets_lidar[k][7] = 1
            dets_lidar[k][8] = 2  #sensor type: 2

            q_lidar = Quaternion(axis=[0, 0, -1], angle=dets_lidar[k][4])
            T_lidar = q_lidar.transformation_matrix
            T_lidar[0][3] = dets_lidar[k][1]
            T_lidar[1][3] = dets_lidar[k][2]

            T2 = np.matmul(T1, T_lidar)
            q8d = Quaternion(matrix=T2)

            dets_lidar[k][1] = T2[0][3]
            dets_lidar[k][2] = T2[1][3]
            dets_lidar[k][4] = q8d.radians

            k += 1

        if frame_name ==1:
            seq_dets_lidar = dets_lidar
        else:
            seq_dets_lidar = np.concatenate((seq_dets_lidar, dets_lidar), axis=0)

        dets_cam = np.zeros([len(det_cam), 4])

        # ## TODO : Camera part when i'm done w Radar & Lidar part
        for j in range(len(det_cam)):
            dets_cam[h][0] = frame_name
            cam_x = det_cam[j].get('relative_coordinates').get('center_x') -0.5
            theta = np.arctan(cam_x /fx)

            if cam_x < 0.5:
                theta = - theta

            dets_cam[h][3] = 3  #SENSOR TYPE = 3
            #Camera a_0 transform , the tf_urdf one!!
            Bus_camera = np.array([[8.660], [-0.030], [0.1356]])
            q1 = Quaternion(axis=[1, 0, 0], angle=0.003490659)
            q4 = Quaternion(axis=[1, 0, 0], angle=theta)
            q_camera = q1 * q4
            dets_cam[h][1] = q_camera.radians
            type = det_cam[j].get('class_id')  #class_id = 2 is a car
            dets_cam[h][2] = type
            #TODO what's the class ids??
            print('Camera detected: %s' %(type))
            h += 1

        if frame_name == 1:
            seq_dets_cam = dets_cam
        else:
            seq_dets_cam = np.concatenate((seq_dets_cam, dets_cam), axis=0)


        total_frames += 1
        additional_info = np.zeros([len(dets_lidar), 7])
        additional_info[:,1] = 2
        dets_all = {'dets_lidar': dets_lidar[:,1:8], 'dets_radar':dets_radar[:,1:4], 'info': additional_info, 'seq_dets_pose': seq_dets_pose, 'dets_cam': dets_cam}
        start_time = time.time()
        trackers = mot_tracker.update(dets_all)
        cycle_time = time.time() - start_time
        total_time += cycle_time
#
        for d in trackers:
            bbox3d_tmp = d[0:7]
            id_tmp = d[7]
            ori_tmp = d[8]
            type_tmp = det_id2str[d[9]]
            bbox2d_tmp_trk = d[10:14]
            conf_tmp = d[14]

            # str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame_name, id_tmp,
            #                                                                           type_tmp, ori_tmp,
            #                                                                           bbox2d_tmp_trk[0],
            #                                                                           bbox2d_tmp_trk[1],
            #                                                                           bbox2d_tmp_trk[2],
            #                                                                           bbox2d_tmp_trk[3],
            #                                                                           bbox3d_tmp[0], bbox3d_tmp[1],
            #                                                                           bbox3d_tmp[2], bbox3d_tmp[3],
            #                                                                           bbox3d_tmp[4], bbox3d_tmp[5],
            #                                                                           bbox3d_tmp[6],
            #                                                                           conf_tmp)

            str_to_srite = '%d, %d, %s, 0, 0, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n' % (frame_name, id_tmp,
                                                                                      type_tmp, ori_tmp,
                                                                                      bbox2d_tmp_trk[0],
                                                                                      bbox2d_tmp_trk[1],
                                                                                      bbox2d_tmp_trk[2],
                                                                                      bbox2d_tmp_trk[3],
                                                                                      bbox3d_tmp[0], bbox3d_tmp[1],
                                                                                      bbox3d_tmp[2], bbox3d_tmp[3],
                                                                                      bbox3d_tmp[4], bbox3d_tmp[5],
                                                                                      bbox3d_tmp[6],
                                                                                      conf_tmp)

            print(str_to_srite)
            eval_file.write(str_to_srite)
            print('Check')

    eval_file.close()

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))



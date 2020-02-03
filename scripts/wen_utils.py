#!/usr/bin/env python
# coding: utf-8

'''
 file for utils for CHRISTINAAAAAAAAA
'''

import json
import numpy as np

STATE_SIZE = 14
MEAS_SIZE = 7   #measurement model for pixor, 7
MEAS_SIZE_Radar = 3  #measurement model for radar, 3
#MOTION_MODEL = "CYRA"
MOTION_MODEL="CA"
#MOTION_MODEL="CV"

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
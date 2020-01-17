# WORKING KF for RADAR

from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
from random import randint

from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pyquaternion import Quaternion


plt.ion()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

#TODO update the bounds??
plt.xlim([355000, 355100])
plt.ylim([149850, 149950])
plt.grid(True)

for i in range(1):
    ax = fig.add_subplot(111, aspect='equal')
    plt.grid(True)

    #PRINT EGO POSE VALUE
    k = i +750
    frame = pose[k]
    [counter, header] = [frame.get('counter'),frame.get('header')]
    ts = header.get('stamp')
    print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    egoPose = frame.get('pose')
    [egoCoor, attitude] = [egoPose.get('position'), egoPose.get('attitude')]
    [egoX,egoY] = [egoCoor.get('x'), egoCoor.get('y')]
    plt.xlim([egoX-30, egoX+60])
    plt.ylim([egoY-30, egoY+60])
    yaw = attitude.get('yaw')
    sidex = egoX + (width / 2)
    sidey = egoY + (length / 2)

    #TODO the yaw this??
    yaw2= float(yaw)
    degy = np.rad2deg(yaw2)

    #READ RADAR POINTS
    frameR = dataR[k]
    radarDetection = frameR.get('front_esr_tracklist')
    numObj = len(radarDetection)
    [counterR, headerR] = [frameR.get('counter'), frameR.get('header')]
    tsR = header.get('stamp')
    print(datetime.utcfromtimestamp(tsR).strftime('%Y-%m-%d %H:%M:%S'))

    qWB = Quaternion(axis=[0, 0, 1], angle=yaw2)  # Rz = 0.998583 , yaw of bus
    Twb = qWB.transformation_matrix
    Twb[0, 3] = egoX
    Twb[1, 3] = egoY
    Twb[2, 3] = 1

    # Bus into world frame
    for j in range(numObj):
        objectA = radarDetection[j]
        rangeA = objectA.get('range')
        range_rate = objectA.get('range_rate ')
        angle_centroid = float(objectA.get('angle_centroid'))
        angle_centroid = float(angle_centroid) * (np.pi / 180)
        RPx = rangeA * np.cos(angle_centroid)
        RPy = rangeA * np.sin(angle_centroid)

        # CONVERT POINT INTO WORLD FRAME
         #posRP = np.array([[34.967, -2.2788, 1, 1]]).transpose()  # TODO THE READINGS OF THE measurement
        posRP = np.array([[RPx, RPy, 1, 1]]).transpose()  # TODO THE READINGS OF THE measurement


        #Radar into bus frame
        q1 = Quaternion(axis=[1, 0, 0], angle=-0.00349066)  # Rdar x : -0.00349066
        q2 = Quaternion(axis=[0, 1, 0], angle=-0.00872665)  # Rdar y:  -0.00872665
        q3 = Quaternion(axis=[0, 0, 1], angle= 0)  # Rdar z: 0

        #tf = [8.69, 0, 0.171]
        qBR = q1 * q2 * q3  # radar frame in bus frame
        #qBR = q3 * q2 * q1
        Tbr = qBR.transformation_matrix
        Tbr[0, 3] = 8.69  # 8.69
        Tbr[2, 3] = 0.171  # 0.171

        Twr = np.matmul(Twb, Tbr)

        posR3 = np.matmul(Twr, posRP)
        radarPx = posR3[0, 0]
        radarPy = posR3[1, 0]

        patch3 = ax.add_patch(patches.Circle((radarPx, radarPy), 1, alpha=0.2, facecolor='green',
                             label='Label'))

    posRPoint = np.array([[34.967, -2.2788, 1, 1]]).transpose()
    posR3_const = np.matmul(Twr, posRPoint)
    radarPx2 = posR3_const[0, 0]
    radarPy2 = posR3_const[1, 0]

    ## PLOT POINTS in world frame
    patch = ax.add_patch(patches.Rectangle((sidex, sidey), length,width, angle=degy, alpha=0.2, facecolor='red',
                                            label='Label'))
    patchTarget = ax.add_patch(patches.Circle((radarPx2, radarPy2), width, alpha=0.2, facecolor='blue',
                                     label='Label'))



    #if counter % 2 == 0:  # collect results every 10 frames
    ## TO PRINT THE PLOT
    plt.savefig('/home/wen/PycharmProjects/Results/T' + str(counter) + '.png')
    plt.close(fig)
    fig = plt.figure()

plt.close(fig)
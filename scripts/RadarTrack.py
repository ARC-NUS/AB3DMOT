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
#import kf_book.ekf_internal as ekf_internal

dim_x = 14
dim_z = 3

def HJradar(x):

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

    return d                         #range rangerate theta

def hxRadar(x):
    #d = []
    #temp = 2
    range = np.sqrt(x[0][0]**2 + x[1][0]**2)
    rangerate = (x[0][0]*x[8][0] + x[1][0]*x[9][0]) / range

    #if x[0][0] > 0:
    #    temp = x[1][0] / x[0][0]

    theta = np.arctan(x[3])

    # d = np.arange(3).reshape((dim_z, 1))
    #d = np.array([range[0], rangerate[0], theta[0]]).reshape((dim_z, 1))
    return array ([[range, rangerate, theta]]).reshape((dim_z, 1))   #range rangerate theta

class RadarSim(object):
    """ Simulates the radar signal returns from an object
    flying at a constant altitude and velocity in 1D.
    """

    def __init__(self, dt, posx, posy, posz, theta, length, width, height, velx, vely, velz, accx, accy, accz, thetav):
        self.posx = posx
        self.posy = posy
        self.posz = posz
        self.velx = velx
        self.vely = vely
        self.velz = velz
        self.width = width
        self.length = length
        self.height = height
        self.accx = accx
        self.accy = accy
        self.accz = accz
        self.theta = theta
        self.thetav = thetav

        self.dt = dt


dt = 0.05
rk = ExtendedKalmanFilter(dim_x, dim_z)

# INITIALISATION #

## TRANSFORM and y into the bus frame then into the world frame

#array([[43.65559915], [-2.27878612],  [ 0.48409509],   [ 1.        ]])

posx = 34.967
posy = -2.2788
posz = 1
theta = 0.998583
length = 12
width = 3
height = 1
velx = 1
vely = 1
velz = 1
accx = 1
accy = 1
accz = 1
thetav = 0


rk.x = RadarSim(dt, posx, posy, posz, theta, length, width, height, velx, vely, velz, accx, accy, accz, thetav)

pathJson = '/home/wen/JI_ST-cloudy-day_2019-08-27-21-55-47/set_1/fused_pose/fused_pose_new.json'
with open(pathJson,"r") as json_file:
        pose = json.load(json_file)
        pose = pose.get('ego_loc')
        sizePData = len(pose)

## Read Radar JSON file
pathJson = '/home/wen/JI_ST-cloudy-day_2019-08-27-21-55-47/set_1/radar_obstacles/radar_obstacles.json'
with open(pathJson,"r") as json_file:
        dataR = json.load(json_file)
        dataR = dataR.get('radar')

## TODO : Get the time stamp,
#1. Transform Radar o/p to ego position
#2. Then update

plt.ion()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
plt.xlim([355000, 355100])
plt.ylim([149850, 149950])
plt.grid(True)


# UPDATE STEPS
# make an imperfect starting guess
#rk.x = array([radar.posx - 100, radar.velx + 100, radar.accx + 1000])
#rk.x = array([radar.posx, radar.posy, radar.velx, radar.accx])
rk.x = np.arange(14).reshape((14, 1))

rk.F = eye(dim_x)
rk.F[0,7] = rk.F[1,8]= rk.F[2,9] = dt

range_std = 5.  # meters
rk.R = np.diag([range_std ** 2])*eye(dim_z)
#rk.Q[0:dim_x-1, 0:dim_x-1] = Q_discrete_white_noise(2, dt=dt, var=0.1)
rk.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
rk.Q[2, 2] = 0.1
rk.P *= 0

xs, track = [], []

#sqrt(34.967**2 + (-2.2788)**2)

for i in range(int(1 / dt)): #20
    #track.append((radar.posx, radar.velx, radar.theta))
    #z = array([[1],[1]])
    #z = eye(dim_z, 1)
    #z = np.arange(3).reshape((3,1))
    z = np.array([35.041, 0.1, 1]) # range, range rate, angle centroid THESE VALUES MUST BE UPDATED WITH THE READING ONES!!
    z = z.reshape((3, 1))
    #z = array([[1], [2], [3]])
    rk.update(z, HJradar, hxRadar)
    xs.append(rk.x)
    rk.predict()


np.diag(rk.x*eye(14))

xs = asarray(xs)
track = asarray(track)
time = np.arange(0, len(xs) * dt, dt)

#ekf_internal.plot_radar(xs, track, time)

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
    plt.savefig('/home/wen/AB3DMOT/Visual_results/' + str(counter) + '.png')
    plt.close(fig)
    fig = plt.figure()

plt.close(fig)
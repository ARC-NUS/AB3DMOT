
import cv2
import numpy as np

width = 416
height = 416


camera_matrix = np.eye(3)
camera_matrix[0, 2] = width / 2.0  # define center x
camera_matrix[1, 2] = height / 2.0  # define center y
camera_matrix[0, 0] = 958.5928517660333  # define focal length x
camera_matrix[1, 1] = 961.1122866843237  # define focal length y

distCoeff = np.zeros((4, 1), np.float64)

k1 = -0.9
k2 = 0.61
p1 = 0
p2 = 0

distCoeff[0, 0] = k1;
distCoeff[1, 0] = k2;
distCoeff[2, 0] = p1;
distCoeff[3, 0] = p2;

## (u,v) is the input point, (u', v') is the output point
## camera_matrix=[fx 0 cx; 0 fy cy; 0 0 1]
## P=[fx' 0 cx' tx; 0 fy' cy' ty; 0 0 1 tz]

pts_uv = np.array([0.749004, 0.472985])
num_pts = pts_uv.size / 2
pts_uv.shape = (int(num_pts), 1, 2)
pts_uv = np.array(pts_uv)
pts_uv_undistorted = cv2.undistortPoints(pts_uv, camera_matrix, distCoeff)
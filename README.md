# A Baseline for 3D Multi-Object Tracking 

<img align="center" src="https://github.com/xinshuoweng/AB3DMOT/blob/master/github_demo.gif">

This repository contains the official python implementation for "[A Baseline for 3D Multi-Object Tracking](https://arxiv.org/pdf/1907.03961.pdf)". If you find this code useful, please cite our paper:

```
@article{Weng2019_3dmot, 
  archivePrefix = {arXiv}, 
  arxivId = {1907.03961}, 
  author = {Weng, Xinshuo and Kitani, Kris}, 
  eprint = {1907.03961}, 
  journal = {arXiv:1907.03961}, 
  title = {{A Baseline for 3D Multi-Object Tracking}}, 
  url = {https://arxiv.org/pdf/1907.03961.pdf}, 
  year = {2019} 
}
```
## Overview
- [News](#news)
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [3D Object Detection](#3d-object-detection)
- [3D Multi-Object Tracking](#3d-multi-object-tracking)
- [Acknowledgement](#acknowledgement)

## News
- Aug. 21, 2019: Python 3.5 (3.6?, 3.7?) supported.
- Aug. 21, 2019: Results on KITTI "pedestrian" and "cyclist" categories released.
- Aug. 19, 2019: A minor bug in orientation correction fixed.
- Jul. 9, 2019: Code and results on KITTI "car" category released.

## Introduction
3D multi-object tracking (MOT) is an essential component technology for many real-time applications such as autonomous driving or assistive robotics. However, recent works for 3D MOT tend to focus more on developing accurate systems giving less regard to computational cost and system complexity. In contrast, this work proposes a simple yet accurate real-time baseline 3D MOT system. We use an off-the-shelf 3D object detector to obtain oriented 3D bounding boxes from the LiDAR point cloud. Then, a combination of 3D Kalman filter and Hungarian algorithm is used for state estimation and data association. Although our baseline system is a straightforward combination of standard methods, we obtain the state-of-the-art results. To evaluate our baseline system, we propose a new 3D MOT extension to the official KITTI 2D MOT evaluation along with two new metrics. Our proposed baseline method for 3D MOT establishes new state-of-the-art performance on 3D MOT for KITTI, improving the 3D MOTA from 72.23 of prior art to 76.47. Surprisingly, by projecting our 3D tracking results to the 2D image plane and compare against published 2D MOT methods, our system places 2nd on the official KITTI leaderboard. Also, our proposed 3D MOT method runs at a rate of 214.7 FPS, 65 times faster than the state-of-the-art 2D MOT system. 

## Dependencies:
This code has been tested on python 2.7 and 3.5, and also requires the following packages:
1. scikit-learn==0.19.2
2. filterpy==1.4.5
3. numba==0.43.1
4. matplotlib==2.2.3
5. pillow==5.2.0
6. opencv-python==3.4.3.18
7. glob2==0.6
8. pypcd for pixor stuff
9. munkres==1.0.12
10. shapely>=1.6.4

One can either use the system python or create a virtual enviroment (virtualenv for python2, venv for python3) specifically for this project (https://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv). To install required dependencies on the system python, please run the following command at the root of this code:
```
$ pip2 install -r requirements.txt
```
To install required dependencies on the virtual environment of the python (e.g., virtualenv for python2), please run the following command at the root of this code:
```
$ pip install virtualenv
$ virtualenv .
$ source bin/activate
$ pip install -r requirements.txt
```

## 3D Object Detection:
For convenience, we provide the 3D detection of the PointRCNN on the KITTI MOT dataset at (./data/KITTI/) for car, pedestrian and cyclist splits. 

## 3D Multi-Object Tracking

### Inference
To run our tracker on the KITTI MOT validation set with the provided detection:

```
$ python main.py car_3d_det_val
$ python main.py ped_3d_det_val
$ python main.py cyc_3d_det_val
```
To run our tracker on the KITTI MOT test set with the provided detection:

```
$ python main.py car_3d_det_test
$ python main.py ped_3d_det_test
$ python main.py cyc_3d_det_test
```
Then, the results will be saved to ./results folder. Note that, please run the code when the CPU is not occupied by other programs otherwise you might not achieve similar speed as reported in our paper.

### 3D MOT Evaluation

To reproduce the quantitative results of our 3D MOT system using the proposed KITTI-3DMOT evaluation tool, please run:
  ```
  $ python evaluation/evaluate_kitti3dmot.py car_3d_det_val
  $ python evaluation/evaluate_kitti3dmot.py ped_3d_det_val
  $ python evaluation/evaluate_kitti3dmot.py cyc_3d_det_val
  ```
Then, the results should be exactly same as below, except for the FPS which depends on the individual machine. Note that the results for car are a little bit better than results in the paper. Also, we add results on pedestrian and cyclist which are not present in the paper.

 Category       | AMOTA (%) | AMOTP (%) | MOTA (%) | MOTP (%)| MT (%) | ML (%) | IDS | FRAG | FPS 
--------------- |:---------:|:---------:|:--------:|:-------:|:------:|:------:|:---:|:----:|:---:
 *Car*          | 39.48     | 74.67     | 76.57    |  79.16  |  70.04 | 7.27   |  0  | 50   | 207.4
 *Pedestrian*   | 25.21     | 49.69     | 61.19    |  67.00  |  36.53 | 39.52  |  0  | 63   | 436.6
 *Cyclist*      | 20.05     | 59.29     | 58.47    |  75.25  |  56.76 | 27.03  |  0  | 5    | 1168.5

### Visualization

To reproduce the qualitative results of our 3D MOT system shown in the paper:

1. Thresholding the trajectories using a proper threshold
2. draw the remaining 3D trajectories on the images (Note that the opencv3 is required by this step, please check the opencv version if there is an error)
  ```
  $ python trk_conf_threshold.py car_3d_det_test
  $ python visualization.py car_3d_det_test_thres
  ```

Then, the visualization results are saved to ./results/car_3d_det_test_thres/trk_image_vis. If one wants to visualize the results on the entire sequences, please first download the KITTI MOT dataset at http://www.cvlibs.net/datasets/kitti/eval_tracking.php and move the image and calibration files to the './data/KITTI/resources' folder.

In addition, one can check out our demo for viusualization in full_demo.mp4


### 2D MOT Evaluation
To reproduce the quantitative results of our 3D MOT system using the official KITTI 2D MOT evaluation server for car category shown in the paper, please compress the folder below and upload to http://www.cvlibs.net/datasets/kitti/user_submit.php
  ```
  $ ./results/car_3d_det_test_thres/data
  ```
Then, the results should be similar to our entry on the KITTI 2D MOT leaderboard: 

 Category       | MOTA (%) | MOTP (%)| MT (%) | ML (%) | IDS | FRAG | FPS 
--------------- |:--------:|:-------:|:------:|:------:|:---:|:----:|:---:
 *Car*          | 83.84    |  85.24  | 66.92  | 11.38  |  9  | 224  | 214.7
 
 
## Acknowledgement
Part of the code is borrowed from "[SORT](https://github.com/abewley/sort)"

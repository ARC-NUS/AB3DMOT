
from __future__ import print_function
import os.path, copy, numpy as np, time, sys
import json
from datetime import datetime
from KFTracking import happyTracker
from wen_utils import readJson
from check_iou_jsons import check_iou_json
import glob
import csv


if __name__ == '__main__':
    print("Initialising...")

    det_id2str = {0: 'Pedestrian', 2: 'Car', 3: 'Cyclist', 4: 'Motorcycle', 5: 'Truck'}

    # TODO : Change the base directory!!!

    # bag_dir = "/home/wen/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/10_jan/log_high/"
    # basedir_total = sorted(glob.glob(bag_dir + "set*"))
    basedir_total  =  ['/media/wen/demo_ssd/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/16_sep/log_high/set_8',
                        '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_3',
                        '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_2',
                        '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_1',
                        '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_12',
                        '/media/wen/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_3',
                        '/media/wen/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_9']
    labels_total =  ['/media/wen/demo_ssd/raw_data/train_labels/JI_ST-cloudy-day_2019-08-27-21-55-47/set_8',
                        '/media/wen/demo_ssd/raw_data/train_labels/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/set_3',
                        '/media/wen/demo_ssd/raw_data/train_labels/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/set_2',
                        '/media/wen/demo_ssd/raw_data/train_labels/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/set_1',
                        '/media/wen/demo_ssd/raw_data/train_labels/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/set_12',
                        '/media/wen/demo_ssd/raw_data/train_labels/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/set_3',
                        '/media/wen/demo_ssd/raw_data/train_labels/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/set_9']
    num_testcase = 3**20

    names = ['Set Number', 'Max age', 'Min hits', 'Hung thres',  'rlA', 'rlB', 'rlC',
             'rlD', 'rlE', 'rlF', 'rlG', 'rlH', 'rlJ', 'rlK', 'rlL', 'rlM',
             'AMOTA','AMOTP']
    MOT_total = names


    MOT_curr = np.zeros([1, 20])
    hung_thresh_total = np.array([0.01, 0.03, 0.05])
    rng_thres = np.array([1,10,100])
    # tracker_json_outfile = basedir +  "/TrackOutput_Set" + set_num + '_' + d1 + ".json"

     # CURRENT: Do a coarse grid search
    for i in range(len(basedir_total)):
        basedir = basedir_total[i]
        print(basedir)
        labels_json_path = glob.glob(labels_total[i] + "/*annotations.json")
        print(labels_json_path)
        # Join various path components
        pathRadar = os.path.join(basedir, "radar_obstacles/radar_obstacles.json")
        pathCamera_a0 = glob.glob(basedir + "/image_detections/results_a0*.json")[0]
        pathCamera_a3 = glob.glob(basedir + "/image_detections/results_a3*.json")[0]
        pathLidar = basedir + '/pixor_outputs_pixorpp_kitti_nuscene_stk.json'
        pathIBEO = basedir + '/ecu_obj_list/ecu_obj_list.json'
        pathPose = basedir + '/fused_pose/fused_pose.json'

        dataR, dataL, dataC, dataC_a3, dataPose, dataIB = readJson(pathRadar, pathLidar, pathCamera_a0, pathCamera_a3,
                                                                   pathPose, pathIBEO)
        if i == 0:
            dataR_total = dataR
            dataL_total = dataL
            dataC_total = dataC
            dataC_a3_total = dataC_a3
            dataPose_total = dataPose
            dataIB_total = dataIB
        else:
            dataR_total = np.vstack((dataR_total, dataR))
            dataL_total = np.vstack((dataL_total, dataL))
            dataC_total = np.vstack((dataC_total, dataC))
            dataC_a3_total = np.vstack((dataC_a3_total, dataC_a3))
            dataPose_total =  np.vstack((dataPose_total, dataPose))
            dataIB_total = np.vstack((dataIB_total, dataIB))

    #     #FIXME : Do a coarse grid search
#    for max_age in range(1,5):
    count = 0
rlA
    tracker_json_outfile = '/home/wen/AB3DMOT/scripts/results/sensorfusion/GSearch_20200319.json'
    savePath = '/home/wen/AB3DMOT/scripts/results/sensorfusion/GSearch_20200319.csv'

    myFile = open(savePath, 'a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(names)


    for rlA in range(len(rng_thres)):
        for rlB in range(len(rng_thres)):
            for rlC in range(len(rng_thres)):
                for rlD in range(len(rng_thres)):
                    for max_age in range(2, 7):
                        for min_hits in range(2, 7):
                            for ht in range(len(hung_thresh_total)):
                    # for rlE in range(len(rng_thres)):
                    #     for rlF in range(len(rng_thres)):
                    #         for rlG in range(len(rng_thres)):
                    #             for rlH in range(len(rng_thres)):
                    #                 for rlJ in range(len(rng_thres)):
                    #                     for rlK in range(len(rng_thres)):
                    #                         for rlL in range(len(rng_thres)):
                    #                             for rlM in range(len(rng_thres)):

                                AMOTA = 0
                                AMOTP = 0

                                for i in range(len(basedir_total)):
                                    dataR= dataR_total[i]; dataL = dataL_total[i];  dataC = dataC_total[i];
                                    dataC_a3 = dataC_a3_total[i]; dataPose= dataPose_total[i]; dataIB = dataIB_total[i];
                                    testCamDar = 1; testPIXOR = 1; testIBEO = 1;

                                    isReady = 1
                                    basedir = basedir_total[i]
                                    print(basedir)
                                    labels_json_path = glob.glob(labels_total[i] + "/*annotations.json")

                                    count = count +1
                                    print (count)

                                    if isReady == 1:
                                        hung_thresh = hung_thresh_total[ht]

                                        Rlidar = np.identity(7)
                                        Rlidar[2, 2] = 10. ** -5 # z
                                        Rlidar[6, 6] = 10. ** -5 # h

                                        Qmodel = np.identity(14)
                                        #tuning
                                        Qmodel[0][0] *= rng_thres[rlA]
                                        Qmodel[1][1] = Qmodel[0][0]
                                        Qmodel[3][3] *= rng_thres[rlB]
                                        Qmodel[4][4] *= rng_thres[rlC]
                                        Qmodel[5][5] =Qmodel[4][4]
                                        Qmodel[7][7] *= rng_thres[rlD]
                                        Qmodel[8][8] =Qmodel[7][7]

                                        P_0lidar = np.identity(14)
                                        # #tuning
                                        # P_0lidar[0][0] *= rng_thres[rlE]
                                        # P_0lidar[1][1] = P_0lidar[0][0]
                                        # P_0lidar[3][3] *= rng_thres[rlF]
                                        # P_0lidar[4][4] *= rng_thres[rlG]
                                        # P_0lidar[5][5] = P_0lidar[4][4]
                                        # P_0lidar[7][7] *= rng_thres[rlH]
                                        # P_0lidar[8][8] =P_0lidar[7][7]

                                        Rcr = np.identity(7)
                                        Rcr[2, 2] = 10. ** -5 # z
                                        Rcr[6, 6] = 10. ** -5 # h

                                        P_0cr = np.identity(14)

                                        # tuning
                                        # P_0cr[0][0] *= rng_thres[rlJ]
                                        # P_0cr[1][1] = P_0cr[0][0]
                                        # P_0cr[3][3] *= rng_thres[rlK]
                                        # P_0cr[7][7] *= rng_thres[rlH]
                                        # P_0cr[8][8] = P_0cr[7][7]

                                        Ribeo = np.identity(7)
                                        Ribeo[2, 2] = 10. ** -5 # z
                                        Ribeo[6, 6] = 10. ** -5 # h

                                        P_0ibeo = np.identity(14)
                                        # # tuning
                                        # P_0ibeo[0][0] *= rng_thres[rlL]
                                        # P_0ibeo[1][1] = P_0ibeo[0][0]
                                        # P_0ibeo[3][3] *= rng_thres[rlM]
                                        # P_0ibeo[4][4] *= rng_thres[rlG]
                                        # P_0ibeo[5][5] = P_0ibeo[4][4]
                                        # P_0ibeo[7][7] *= rng_thres[rlH]
                                        # P_0ibeo[8][8] = P_0ibeo[7][7]

                                        radarCam_threshold = 0.05  # .05 #radians!!
                                        radar_offset = 0

                                        total_list = happyTracker(dataR, dataL, dataC, dataC_a3,
                                                                  dataPose, dataIB, max_age, min_hits,
                                                                  hung_thresh,
                                                                  Rlidar, Qmodel, P_0lidar, Rcr, P_0cr,
                                                                  Ribeo, P_0ibeo, radarCam_threshold,
                                                                  radar_offset, testPIXOR, testIBEO,
                                                                  testCamDar)

                                        isPrint = 1
                                        isCheckIOU = 1

                                        if isPrint == True:
                                            today = datetime.today()
                                            d1 = today.strftime("%Y_%m_%d")
                                            set_num = '1'
                                            #tracker_json_outfile = basedir +  "/TrackOutput_Set" + set_num + '_' + d1 + ".json"
                                            with open(tracker_json_outfile, "w+") as outfile:
                                                json.dump(total_list, outfile, indent=1)

                                            print('Saved tracking results as Json')

                                        if isCheckIOU == True:

                                            #labels_json_path = glob.glob(basedir +"/labels/*annotations.json")

                                            distance_metric = "IOU"  # using IOU as distance metric
                                            thres_d = 100.  # 100 threshold distance to count as a correspondance, beyond it will be considered as missed detection

                                            MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = check_iou_json(labels_json_path[0],
                                                                                                                                         tracker_json_outfile,
                                                                                                                                         thres_d,
                                                                                                                                         distance_metric)
                                            print (MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt)

                                            #names = ['Set Number', 'Max age', 'Min hits', 'Hung thres', 'MOTA', 'MOTP']
                                            #row = num_testcase*i + num

                                            AMOTA = AMOTA + MOTA
                                            AMOTP = AMOTP + MOTP

                                MOT_curr[0][0] = int(basedir[-1])
                                MOT_curr[0][1] = max_age
                                MOT_curr[0][2] = min_hits
                                MOT_curr[0][3] = hung_thresh
                                MOT_curr[0][4] = rng_thres[rlA]
                                MOT_curr[0][5] = rng_thres[rlB]
                                MOT_curr[0][6] = rng_thres[rlC]
                                MOT_curr[0][7] = rng_thres[rlD]
                                # MOT_curr[0][8] = rng_thres[rlE]
                                # MOT_curr[0][9] = rng_thres[rlF]
                                # MOT_curr[0][10] = rng_thres[rlG]
                                # MOT_curr[0][11] = rng_thres[rlH]
                                # MOT_curr[0][12] = rng_thres[rlJ]
                                # MOT_curr[0][13] = rng_thres[rlK]
                                # MOT_curr[0][14] = rng_thres[rlL]
                                # MOT_curr[0][15] = rng_thres[rlM]
                                MOT_curr[0][16] = AMOTA/len(basedir_total)
                                MOT_curr[0][17] = AMOTP/len(basedir_total)

                                myFile = open(savePath, 'a')
                                with myFile:
                                    writer = csv.writer(myFile)
                                    writer.writerows(MOT_curr)
#
    #                                                         MOT_total = np.vstack((MOT_total, MOT_curr))
    #
    # myData = MOT_total #[[1, 2, 3], ['Good Morning', 'Good Evening', 'Good Afternoon']]
    # myFile = open(savePath, 'w')
    # with myFile:
    #     writer = csv.writer(myFile)
    #     writer.writerows(myData)

    print ('Completed Tracking')

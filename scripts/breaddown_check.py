from __future__ import print_function
import os.path, copy, numpy as np, time, sys
import json
from datetime import datetime
from KFTracking import happyTracker
from wen_utils import readJson
from check_iou_jsons import check_iou_json, check_iou_json_class
import glob
import csv
import time
def safe_div(x, y):
    status = 1
    if y == 0:
        return 0.0
    return float(x)/y


if __name__ == '__main__':
    print("Initialising...")




    det_id2str = {0: 'Pedestrian', 2: 'Car', 3: 'Cyclist', 4: 'Motorcycle', 5: 'Truck'}


    basedir_total = ['/media/wen/demo_ssd/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/16_sep/log_high/set_8',
                     '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_3',
                     '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_2',
                     '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_1',
                     '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_12',
                     '/media/wen/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_3',
                     '/media/wen/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_9']

    labels_total = ['/media/wen/demo_ssd/raw_data/trash/train_labels/JI_ST-cloudy-day_2019-08-27-21-55-47/set_8',
                    '/media/wen/demo_ssd/raw_data/trash/train_labels/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/set_3',
                    '/media/wen/demo_ssd/raw_data/trash/train_labels/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/set_2',
                    '/media/wen/demo_ssd/raw_data/trash/train_labels/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/set_1',
                    '/media/wen/demo_ssd/raw_data/trash/train_labels/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/set_12',
                    '/media/wen/demo_ssd/raw_data/trash/train_labels/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/set_3',
                    '/media/wen/demo_ssd/raw_data/trash/train_labels/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/set_9']

    num_testcase = 3 ** 20

    names = ['Set Number', 'Max age', 'Min hits', 'Hung thres', 'rlA - Q xy', 'rlB - Q theta', 'rlC - P_0 xy ',
             'rlD - P_0 xy ', 'rlE - P_0 xy ', 'AMOTA', 'AMOTP', 'AMOTA- ped' 'AMOTP-ped', 'AMOTA- bi', 'AMOTP-bi',
             'AMOTA- pmd' 'AMOTP-pmd', 'AMOTA- motorbike', 'AMOTP-motorbike', 'AMOTA- car', 'AMOTP-car',
             'AMOTA- truck', 'AMOTP-truck', 'AMOTA- bus', 'AMOTP-bus']

    MOT_total = names

    MOT_curr = np.zeros([1, 22])
    #hung_thresh_total = np.array([0.01, 0.03])
    rng_thres = np.array([0.01, 0.1, 1, 10, 100])
    # tracker_json_outfile = basedir +  "/TrackOutput_Set" + set_num + '_' + d1 + ".json"

    today = datetime.today()
    d1 = today.strftime("%Y_%m_%d")

    tracker_json_outfile = '/home/wen/AB3DMOT/scripts/results/sensorfusion/csv_wCR/allset_w3CR_' +d1 + '.json'

    # CURRENT: Do a coarse grid search
    for i in range(len(basedir_total)):
        basedir = basedir_total[i]
        print(basedir)
        labels_json_path = glob.glob(labels_total[i] + "/*annotations.json")
        print(labels_json_path)
        # Join various path components
        pathRadar = os.path.join(basedir, "radar_obstacles/radar_obstacles.json")

        #Camera 6 class model :: yolov3-tiny-prn-slim_best.weights
        # pathCamera_a0 = glob.glob(basedir + "/image_detections/results_a0_yolov3-tiny-prn-slim_best.weights*.json")[0]
        # pathCamera_a3 = glob.glob(basedir + "/image_detections/results_a3_yolov3-tiny-prn-slim_best.weights*.json")[0]

        pathCamera_a0 = glob.glob(basedir + "/image_detections/results_cama0*.json")[0]
        pathCamera_a3 = glob.glob(basedir + "/image_detections/results_cama3*.json")[0]

        #pathLidar = basedir + '/pixor_outputs_pixorpp_kitti_nuscene_stk.json'
        pathLidar = basedir + '/pixor_outputs_tf_epoch_150_valloss_0.2106.json'
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
            dataPose_total = np.vstack((dataPose_total, dataPose))
            dataIB_total = np.vstack((dataIB_total, dataIB))


    count = 0

    Test_v = 15817


    HAMOTA_all =   0.431070974899
    HAMOTP_all = 87.10681218624508
    HC_all = 12569

    HAMOTA_vehicles = 0.5132928917168759
    HAMOTP_vehicles =  87.81524986706007
    HCv = 12566

    #
    # HAMOTA_all =   0.431070974899
    # HAMOTP_all = 87.10681218624508
    # HC_all = 12569
    #
    # HAMOTA_vehicles = 0.5132928917168759
    # HAMOTP_vehicles =  87.81524986706007
    # HCv = 12566

    HAMOTA_ped = -10
    HAMOTP_ped = 0
    HCp = 0
    num_class = 8

    for rlA in range(len(rng_thres)):
        for rlB in range(len(rng_thres)):
            for rlC in range(len(rng_thres)):
                for rlD in range(len(rng_thres)):
                    for rlE in range(len(rng_thres)):
                        for max_age in range(3, 8):
                            for min_hits in range(3, 8):
                                # for ht in range(len(hung_thresh_total)):
                                AMOTA = 0
                                AMOTP = 0
                                AMOTAclass =np.zeros(num_class)
                                AMOTPclass =np.zeros(num_class)
                                sum_MOTAclass = np.zeros(num_class)
                                sum_MOTPclass = np.zeros(num_class)
                                sum_MOTAstatus = np.zeros(num_class)
                                sum_MOTPstatus = np.zeros(num_class)

                                count = count + 1
                                #if count == 8681 or count > 8643:
                                #if count > 12551 or count < 12575 :

                                start_time = time.time()

                                if count == Test_v:
                                    print(count)
                                    for i in range(len(basedir_total)):
                                        print(
                                            'Beginning tracking: with All sensors, camera is at 10Hz, Lidar at 20Hz, IBEO at 20Hz??, tracker output at 20Hz.. ..')
                                        dataR = dataR_total[i];
                                        dataL = dataL_total[i];
                                        dataC = dataC_total[i];
                                        dataC_a3 = dataC_a3_total[i];
                                        dataPose = dataPose_total[i];
                                        dataIB = dataIB_total[i];

                                        testCamDar = 1;
                                        testPIXOR = 1;
                                        testIBEO = 1;

                                        isReady = 1
                                        basedir = basedir_total[i]
                                        # print(basedir)
                                        labels_json_path = glob.glob(labels_total[i] + "/*annotations.json")

                                        # count = count +1
                                        # print (count)

                                        if isReady == 1:
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
                                            Rcr[0, 0] = 0.001  #error in x and y !! for camera radar fusion
                                            Rcr[1, 1] = 0.001 #error in x and y !! for camera radar fusion
                                            Rcr[2, 2] = 10. ** -5  # z
                                            Rcr[6, 6] = 10. ** -5  # h

                                            P_0cr = np.identity(14)

                                            # tuning
                                            P_0cr[0][0] *= rng_thres[rlD]
                                            P_0cr[1][1] = P_0cr[0][0]
                                            # P_0cr[3][3] *= rng_thres[rlD]
                                            # P_0cr[7][7] *= rng_thres[rlH]
                                            # P_0cr[8][8] = P_0cr[7][7]

                                            Ribeo = np.identity(7)
                                            Ribeo[0, 0] = 0.01  # 10cm 0.1*0.1   0.01
                                            Ribeo[1, 1]  = 0.01 #10cm 0.1*0.1   0.01
                                            Ribeo[2, 2] = 10. ** -5  # z
                                            Ribeo[6, 6] = 10. ** -5  # h

                                            P_0ibeo = np.identity(14)
                                            # # tuning
                                            P_0ibeo[0][0] *= rng_thres[rlE]
                                            P_0ibeo[1][1] = P_0ibeo[0][0]
                                            # P_0ibeo[3][3] *= rng_thres[rlD]
                                            # P_0ibeo[4][4] *= rng_thres[rlG]
                                            # P_0ibeo[5][5] = P_0ibeo[4][4]
                                            # P_0ibeo[7][7] *= rng_thres[rlH]
                                            # P_0ibeo[8][8] = P_0ibeo[7][7]

                                            radarCam_threshold = 0.1  # .05 #radians!!
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
                                                with open(tracker_json_outfile, "w+") as outfile:
                                                    json.dump(total_list, outfile, indent=1)

                                                # print('Saved tracking results as Json')

                                            if isCheckIOU == True:
                                                #labels_json_path = glob.glob(basedir +"/labels/*annotations.json")

                                                distance_metric = "IOU"  # using IOU as distance metric
                                                thres_d = 100.  # 100 threshold distance to count as a correspondance, beyond it will be considered as missed detection


                                                #OLD TRACKER
                                                # MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet, total_gt = check_iou_json(
                                                #     labels_json_path[0],
                                                #     tracker_json_outfile,
                                                #     thres_d,
                                                #     distance_metric)

                                                #print(MOTA, MOTP, total_dist, total_ct, total_mt, total_fpt, total_mmet,
                                                     # total_gt)

                                                MOTA, MOTP, MOTA_class, MOTP_class, total_missed_class, total_gt_class, total_dist_class, MOTA_status, MOTP_status = check_iou_json_class(
                                                    labels_json_path[0],
                                                    tracker_json_outfile,
                                                    thres_d,
                                                    distance_metric)

                                                # print(MOTA_class, MOTP_class)
                                                AMOTA += MOTA
                                                AMOTP += MOTP

                                                sum_MOTAclass += MOTA_class
                                                sum_MOTPclass += MOTP_class
                                                sum_MOTAstatus += MOTA_status
                                                sum_MOTPstatus += MOTP_status

                                    for i in range(5):
                                        AMOTAclass[i] = safe_div(sum_MOTAclass[i], sum_MOTAstatus[i])
                                        #print (sum_MOTAclass, sum_MOTAstatus, sum_MOTPclass, sum_MOTPstatus)
                                        AMOTPclass[i] = safe_div(sum_MOTPclass[i], sum_MOTAstatus[i])

                                    MOT_curr[0][0] = count
                                    MOT_curr[0][1] = max_age
                                    MOT_curr[0][2] = min_hits
                                    MOT_curr[0][3] = hung_thresh
                                    MOT_curr[0][4] = rng_thres[rlA]
                                    MOT_curr[0][5] = rng_thres[rlB]
                                    MOT_curr[0][6] = rng_thres[rlC]
                                    MOT_curr[0][7] = rng_thres[rlD]
                                    MOT_curr[0][8] = rng_thres[rlE]
                                    MOT_curr[0][9] = AMOTA / len(basedir_total)
                                    MOT_curr[0][10] = AMOTP / len(basedir_total)
                                    MOT_curr[0][11] = AMOTAclass[0]
                                    MOT_curr[0][12] = AMOTPclass[0]
                                    MOT_curr[0][13] = AMOTAclass[1]
                                    MOT_curr[0][14] = AMOTPclass[1]
                                    MOT_curr[0][15] = AMOTAclass[2]
                                    MOT_curr[0][16] = AMOTPclass[2]
                                    MOT_curr[0][17] = AMOTAclass[3]
                                    MOT_curr[0][18] = AMOTPclass[3]
                                    MOT_curr[0][19] = AMOTAclass[4]
                                    MOT_curr[0][20] = AMOTPclass[4]


                                    if MOT_curr[0][9] > HAMOTA_all:
                                        HAMOTA_all = MOT_curr[0][9]
                                        HAMOTP_all = MOT_curr[0][10]
                                        HC_all = count

                                    if  AMOTAclass[4] > HAMOTA_vehicles and AMOTPclass[4] != 0:
                                        HAMOTA_vehicles = AMOTAclass[4]
                                        HAMOTP_vehicles = AMOTPclass[4]
                                        HCv = count
                                    print (AMOTAclass[0], AMOTPclass[0])
                                    if AMOTAclass[0] > HAMOTA_ped and AMOTPclass[0] != 0:
                                        HAMOTA_ped = AMOTAclass[0]
                                        HAMOTP_ped = AMOTPclass[0]
                                        HCp = count


                                    print('AMOTA :', MOT_curr[0][9], 'AMOTP :', MOT_curr[0][10], 'Using Camera Radar:', testCamDar)
                                    print('HAMOTA(ALL) :', HAMOTA_all, 'HAMOTP : ', HAMOTP_all, 'Count', HC_all, 'Using Camera Radar:', testCamDar)
                                    print('HAMOTA(vehicles) :', HAMOTA_vehicles, 'HAMOTP : ', HAMOTP_vehicles, 'Count', HCv, 'Using Camera Radar:', testCamDar)
                                    print('HAMOTA(pedesterians) :', HAMOTA_ped, 'HAMOTP : ', HAMOTP_ped, 'Count', HCp, 'Using Camera Radar:', testCamDar)

                                #print("--- %s seconds ---" % (time.time() - start_time))


    print('Completed Tracking')




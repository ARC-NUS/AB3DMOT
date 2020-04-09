from __future__ import print_function
import os.path, copy, numpy as np, time, sys
import json
from datetime import datetime
from KFTracking import happyTracker
from wen_utils import readJson
from check_iou_jsons import check_iou_json, check_iou_json_class
import glob
import csv

def safe_div(x, y):
    status = 1
    if y == 0:
        return 0.0
    return float(x)/y


if __name__ == '__main__':
    print("Initialising...")

    det_id2str = {0: 'Pedestrian', 2: 'Car', 3: 'Cyclist', 4: 'Motorcycle', 5: 'Truck'}

    # #if testing ALL directories
    # set_v = glob.glob("/media/wen/demo_ssd/raw_data/*/*/log_high/set*")
    # sets = []
    # for set in set_v:
    #     if (set[-2:-1] + set[-1]) != "_0":
    #         if sets == []:
    #             sets = set
    #         else:
    #             sets = np.vstack((sets, set))

    basedir_total = ['/media/wen/demo_ssd/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/16_sep/log_high/set_8',
                     '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_3',
                     '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_2',
                     '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_1',
                     '/media/wen/demo_ssd/raw_data/ST_CETRAN-cloudy-day_2019-08-27-22-30-18/sep/log_high/set_12',
                     '/media/wen/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_3',
                     '/media/wen/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_9']

    set_v = glob.glob("/media/wen/demo_ssd/raw_data/*/*/log_high/set*")
    print (set_v)
    sets = []
    for set in set_v:

        if (set[-2:-1] + set[-1]) != "_0" and (set[-28:]) != '11-25/16_sep/log_high/set_13' and (set[-28:]) != '47-10/11_sep/log_high/set_13' and set[-25:] != '30-18/sep/log_high/set_19' :
            if sets == []:
                sets = set
            else:
                sets = np.vstack((sets, set))

    basedir_total = sets

    rng_thres = np.array([0.01, 0.1, 1, 10, 100])


    count = 0

    Test_v = 12566

    HAMOTA_all =   0.431070974899
    HAMOTP_all = 87.10681218624508
    HC_all = 12569

    HAMOTA_vehicles = 0.5132928917168759
    HAMOTP_vehicles =  87.81524986706007
    HCv = 12566

    HAMOTA_ped = -0.00980392156862747
    HAMOTP_ped = 25.0
    HCp = 10255


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
                                AMOTAclass =np.zeros(7)
                                AMOTPclass =np.zeros(7)
                                sum_MOTAclass = np.zeros(7)
                                sum_MOTPclass = np.zeros(7)
                                sum_MOTAstatus = np.zeros(7)
                                sum_MOTPstatus = np.zeros(7)

                                count = count + 1

                                if count < Test_v:
                                    print(count)

                                if count == Test_v :
                                    print (count)
                                    for i in range(len(basedir_total)):

                                        # basedir = basedir_total[i]

                                        # For generating prediction :
                                        basedir = basedir_total[i][0]

                                        print(basedir)
                                        # labels_json_path = glob.glob(labels_total[i] + "/*annotations.json")
                                        # print(labels_json_path)
                                        # Join various path components
                                        pathRadar = os.path.join(basedir, "radar_obstacles/radar_obstacles.json")

                                        # Camera 6 class model :: yolov3-tiny-prn-slim_best.weights
                                        # pathCamera_a0 = glob.glob(basedir + "/image_detections/results_a0_yolov3-tiny-prn-slim_best.weights*.json")[0]
                                        # pathCamera_a3 = glob.glob(basedir + "/image_detections/results_a3_yolov3-tiny-prn-slim_best.weights*.json")[0]

                                        pathCamera_a0 = \
                                        glob.glob(basedir + "/image_detections/results_cama0*.json")[0]
                                        pathCamera_a3 = \
                                        glob.glob(basedir + "/image_detections/results_cama3*.json")[0]

                                        # pathLidar = basedir + '/pixor_outputs_pixorpp_kitti_nuscene_stk.json'
                                        pathLidar = basedir + '/pixor_outputs_mdl_tf_epoch_150_valloss_0.2106.json'
                                        pathIBEO = basedir + '/ecu_obj_list/ecu_obj_list.json'
                                        pathPose = basedir + '/fused_pose/fused_pose.json'

                                        dataR, dataL, dataC, dataC_a3, dataPose, dataIB = readJson(pathRadar,
                                                                                                   pathLidar,
                                                                                                   pathCamera_a0,
                                                                                                   pathCamera_a3,
                                                                                                   pathPose,
                                                                                                   pathIBEO)



                                        testCamDar = 1;
                                        testPIXOR = 1;
                                        testIBEO = 1;

                                        isReady = 1

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
                                            # Qmodel[4][4] *= rng_thres[rlC]
                                            # Qmodel[5][5] =Qmodel[4][4]
                                            # Qmodel[7][7] *= rng_thres[rlD]
                                            # Qmodel[8][8] =Qmodel[7][7]

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
                                                today = datetime.today()
                                                d1 = today.strftime("%Y_%m_%d")
                                                tracker_json_outfile = basedir + "/trackerresults_wCR_" + str(Test_v)+"_" + d1 + ".json"
                                                print (tracker_json_outfile)
                                                with open(tracker_json_outfile, "w+") as outfile:
                                                    json.dump(total_list, outfile, indent=1)

    print('Completed Tracking')

#/usr/bin/env python
import sys
sys.path.insert(1, '/home/yl/bus_ws/src/pixorpp/pixorpp_torch/script')

import tracking_ouput as pxpp_json

if __name__ == "__main__":
    
    # create json file
    model_path = '/home/yl/bus_ws/src/pixorpp/pixorpp_torch/models/2019_10_04/tf_epoch_3_valloss_0.0093.pt'
    pcds_folder_path = '/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep//log_high/set_7/pcds'
    px_json_file = '/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_high/set_7/pixor_outputs_tf_epoch_3_valloss_0.0093_3.json'
    pxpp_json.create_json(model_path, pcds_folder_path, px_json_file) 
    
    # check pxpp stats
    labels_json = '/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/labels/Set_7_annotations.json'
    iou_thresh = 75
    tp, fp, fn, var= check_output_json(px_json_file, labels_json, iou_thresh)
    print tp, fp, fn, var
    print "precision at iou ", iou_thresh, ": ", (tp*100.0)/ (fp+tp)
    print "recall: ", (tp*100.0)/ (fn+tp)
    
    # run tracker
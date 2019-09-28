#/usr/bin/env python

import numpy as np
import check_iou_jsons as check_iou
from shapely.geometry import Polygon
import json

def check_output_json(output_json, labels_json, iou_thresh):

    with open(output_json, "r+") as px_file:
        with open(labels_json, "r+") as l_file:    
            px_data = json.load(px_file, encoding="utf-8")
            l_data =  json.load(l_file, encoding="utf-8")

            tp = 0
            total_labelled = 0
            total_pixor = 0
            missed = 0
            l_i = 0
            
            var = np.zeros(5)

            for i_scene, px_scene in enumerate(px_data):
                l_scene = l_data[i_scene- l_i] # corresp pixor output
                if px_scene['name'] != l_scene['name']:
                    print "Pixor output json and label json does not match"
                    print "pixor pcd: ", px_scene['name']
                    print "lable pcd ", l_scene['name']
#                     raise Exception
                    l_i += 1
                    continue
#                     return False

                total_labelled += len(l_scene['annotations'])
                total_pixor += len(px_scene['objects'])

                for bb_label in l_scene['annotations']:
                    ow = float(bb_label['geometry']['dimensions']['x'])
                    ob = float(bb_label['geometry']['dimensions']['y'])
                    ox_c = float(bb_label['geometry']['position']['x'])
                    oy_c = float(bb_label['geometry']['position']['y'])
                    otheta = float(bb_label['geometry']['rotation']['z'])
                    l_pts = check_iou.get_vertices(ow,ob,ox_c,oy_c,otheta,0,0,1)
                    is_corresp = False
                    iou_max = 0.
                    err_ss = np.zeros(5)
	    	    
                    for bb_pixor in px_scene['objects']:
                        tw = float(bb_pixor['length'])
                        tb = float(bb_pixor['width'])
                        tx_c = float(bb_pixor['centroid'][0])
                        ty_c = float(bb_pixor['centroid'][1])
                        ttheta = float(bb_pixor['heading'])
                        p_pts = check_iou.get_vertices(tw,tb,tx_c,ty_c,ttheta,0,0,1)
                    
                        iou = check_iou.get_iou(l_pts, p_pts)

                        if iou >= iou_thresh:
                            # check that the iou is the best one for a given label
                            if not is_corresp:
                                # dont count if there is another pixor already associated to this label
                                tp += 1
                                is_corresp = True
                            
                            if iou > iou_max:
                                iou_max = iou
                                err_ss[0] = ox_c - tx_c
                                err_ss[1] = oy_c - ty_c
                                err_ss[2] = ow - tw
                                err_ss[3] = ob - tb
                                err_ss[4] = ttheta - otheta
                        
                    if not is_corresp:
                        missed += 1
                    else:
                        # add to the cov
                        var += err_ss ** 2
        if total_labelled != tp + missed:
            print "error!  total_labelled != tp + missed"
            print "total labelled: ", total_labelled
            # return None, None, None

        fp = total_pixor - tp
        
        var = (var*1.0) / tp

    return tp, fp, missed, var



if __name__ == '__main__':
#     pixor_json = "/media/yl/demo_ssd/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/16_sep/log_low/set_1/pixor_outputs.json"
#     labels_json = "/media/yl/downloads/set1_annotations(1).json" 
    # labels_json = "/media/yl/demo_ssd/raw_data/JI_ST-cloudy-day_2019-08-27-21-55-47/16_sep/log_low/set_1/set1_annotations.json"
#     
    pixor_json = '/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/pixor_outputs.json'
    labels_json = '/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_8/Set_8_annotations.json'

#     pixor_json = '/home/yl/bus_ws/src/AB3DMOT/data/JIC/test_cases/mock_pixor_low.json'
#     labels_json = '/home/yl/bus_ws/src/AB3DMOT/data/JIC/test_cases/mock_labels.json'
     
    iou_thresh = 0.75
    tp, fp, missed, var= check_output_json(pixor_json, labels_json, iou_thresh)
    print tp, fp, missed, var
    print "tp at iou ", iou_thresh, ": ", (tp*1.0)/ (missed+tp)
    
    
    pixor_json = '/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/pixor_outputs.json'
    labels_json = '/media/yl/demo_ssd/raw_data/CETRAN_ST-cloudy-day_2019-08-27-22-47-10/11_sep/log_low/set_7/Set_7_annotations.json'

    tp, fp, missed, var= check_output_json(pixor_json, labels_json, iou_thresh)
    print tp, fp, missed, var
    print "tp at iou ", iou_thresh, ": ", (tp*1.0)/ (missed+tp)

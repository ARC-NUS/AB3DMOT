import os, numpy as np, sys, cv2
from PIL import Image
from utils import is_path_exists, mkdir_if_missing, load_list_from_folder, fileparts, random_colors
from kitti_utils import read_label, compute_box_3d, draw_projected_box3d, Calibration

max_color = 30
colors = random_colors(max_color)  # Generate random colors


type_whitelist = ['Car', 'Pedestrian', 'Cyclist']  # to get the score threshold
score_threshold = -10000

width = 960
height = 604

seq_list = ['0000', '0003']


def vis(dataC, data_root, result_root):
    def show_image_with_boxes(img, objects_res, object_gt, calib, save_path, height_threshold=0):
        img2 = np.copy(img)

        for obj in objects_res:
            box3d_pts_2d, _ = compute_box_3d(obj, calib.P)
            color_tmp = tuple([int(tmp * 255) for tmp in colors[obj.id % max_color]])
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=color_tmp)
            text = 'ID: %d' % obj.id
            if box3d_pts_2d is not None:
                img2 = cv2.putText(img2, text, (int(box3d_pts_2d[4, 0]), int(box3d_pts_2d[4, 1]) - 8),
                                   cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=color_tmp)

        img = Image.fromarray(img2)
        img.save(save_path)

    for seq in seq_list:
        image_dir = os.path.join(data_root, 'image_02/%s' % seq)
        calib_file = os.path.join(data_root, 'calib/%s.txt' % seq)
        result_dir = os.path.join(result_root, '%s/trk_withid/%s' % (result_sha, seq))
        save_3d_bbox_dir = os.path.join(result_dir, '../../trk_image_vis/%s' % seq);
        mkdir_if_missing(save_3d_bbox_dir)

        # load the list
        images_list, num_images = load_list_from_folder(image_dir)
        print('number of images to visualize is %d' % num_images)
        start_count = 0

        for i in range(0,100):
            det_cam = dataC[i]
            x = dataC[i]['name'][0:5]
            image_tmp = np.array(Image.open(base_dir + 'a0_decoded/' + x +'.jpg'))
            img_height, img_width, img_channel = image_tmp.shape


            #TODO from here onwards :(((
            result_tmp = os.path.join(result_dir, '%06d.txt' % image_index)  # load the result
            if not is_path_exists(result_tmp):
                object_res = []
            else:
                object_res = read_label(result_tmp)
            print('processing index: %d, %d/%d, results from %s' % (image_index, count + 1, num_images, result_tmp))
            calib_tmp = Calibration(calib_file)  # load the calibration

            object_res_filtered = []
            for object_tmp in object_res:
                if object_tmp.type not in type_whitelist: continue
                if hasattr(object_tmp, 'score'):
                    if object_tmp.score < score_threshold: continue
                center = object_tmp.t
                object_res_filtered.append(object_tmp)

            num_instances = len(object_res_filtered)
            save_image_with_3dbbox_gt_path = os.path.join(save_3d_bbox_dir, '%06d.jpg' % (image_index))
            show_image_with_boxes(image_tmp, object_res_filtered, [], calib_tmp,
                                  save_path=save_image_with_3dbbox_gt_path)
            print('number of objects to plot is %d' % (num_instances))
            count += 1


if __name__ == "__main__":

    base_dir = '/home/wen/raw_data/another_set/some_date/log_high/set_8/'

    labels_loc = base_dir +'no_qc/set8_30_18_annotations_qc1.json'

    with open(labels_loc, "r") as json_file:
        dataC = json.load(json_file)

    data_root = base_dir + 'a0_decoded/'
    result_root = base_dir + 'labels/annotation_visualisation/a0/'

    vis(dataC, data_root, result_root)
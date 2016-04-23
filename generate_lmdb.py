import sys
import project_config as cfg
sys.path.append('eccv/')
sys.path.append(cfg.caffe_path + 'python')

import mpii_parser as parser
# import config_example as cfg
import mpii_pose_util as pose_util
import numpy as np
from random import shuffle
from tqdm import tqdm
import cv2 as cv
import lmdb
import lmdb_util as lu
import json
from os.path import join as attach

import caffe
import random
from mpii_pose_util import find_sub_bbox

map_size = 1099511627776
key_dict = []

def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        print x,y


def data_augment(img, joints, bboxs, flip=False):
    imshow = False
    for bbox in bboxs:
        x_min, y_min = bbox[0:2]
        x_max, y_max = bbox[2:4]
        new_img = np.array(img[y_min:y_max, x_min:x_max,:], copy=True)

        # if imshow:
        #     temp_img = np.array(img, copy=True)
        #     cv.rectangle(temp_img, (x_min,y_min), (x_max, y_max), (225, 0, 0), 2)
        #     for pt in joints:
        #         cv.circle(temp_img, center=(pt[0], pt[1]), radius=1, color=(255,0,0), thickness=10)
        #     cv.imshow("origin", temp_img)


        new_img = cv.resize(new_img, (227, 227))
        new_joints = [[pt[0] - x_min, pt[1] - y_min] for pt in joints]
        ratio_x = 227.0 / float(x_max - x_min)
        ratio_y = 227.0 / float(y_max - y_min)
        new_joints = [[int(pt[0] * ratio_x), int(pt[1] * ratio_y)] for pt in new_joints]
        yield new_img, new_joints

        # if imshow:
        #     show_new_img = np.array(new_img,copy=True)
        #     for pt in new_joints:
        #         cv.circle(show_new_img, center=(pt[0], pt[1]), radius=1, color=(255,0,0), thickness=10)
        #     cv.imshow("new", show_new_img)


        if cfg.augment_flip:
            flip_img = cv.flip(new_img,1)
            temp_flip_joints = [[227 - pt[0], pt[1]] for pt in new_joints]
            flip_joints = temp_flip_joints[3:6] + temp_flip_joints[0:3] + temp_flip_joints[6:]
            yield  flip_img, flip_joints

        #     if imshow:
        #         show_flip_img = np.array(flip_img, copy=True)
        #         for pt in flip_joints:
        #             cv.circle(show_flip_img, center=(pt[0], pt[1]), radius=1, color=(255,0,0), thickness=10)
        #         cv.imshow("new_flip", show_flip_img)
        #
        # if imshow:
        #     cv.waitKey(0)
        #     cv.destroyAllWindows()



def generate(data, folder, prefix="train"):
    img_lmdb_dir = folder + 'lmdb/' + prefix + "_" +"img.lmdb"
    img_dir = cfg.data_path + "img/"
    img_env = lmdb.Environment(img_lmdb_dir, map_size=map_size)
    img_txn = img_env.begin(write=True, buffers=True)

    joint_lmdb_dir = folder + 'lmdb/' + prefix + "_" + 'joint.lmdb'
    joint_env = lmdb.Environment(joint_lmdb_dir, map_size=map_size)
    joint_txn = joint_env.begin(write=True, buffers=True)

    attribute_name = ['button_type', 'cloth_type', 'placket', 'sleeve_type', "collar_type"]
    attribute_lmdb_dir = [folder + 'lmdb/' + prefix + "_" + each + '.lmdb' for each in attribute_name]
    attribute_env = [lmdb.Environment(each, map_size=map_size) for each in attribute_lmdb_dir]
    attribute_txn = [each.begin(write=True, buffers=True) for each in attribute_env]

    #toursour, collar, button_placket
    bbox_name = ["bbox_toursor","bbox_collar", 'bbox_button_placket', "bbox_left_sleeve", "bbox_right_sleeve"]
    bbox_lmdb_dir = [folder + 'lmdb/' + prefix + "_" + each + '.lmdb' for each in bbox_name]
    bbox_env = [lmdb.Environment(each, map_size=map_size) for each in bbox_lmdb_dir]
    bbox_txn = [each.begin(write=True, buffers=True) for each in bbox_env]
    iterate_list = data
    # problem_set = [ "2014813164828515125157",
    #                 "20146310750331491846",
    #                 "20147213274421802954"]
    # iterate_list = {each:dict(data)[each] for each in problem_set}
    # iterate_list = iterate_list.items()
    bbox_json = {}

    # random.shuffle(iterate_list)
    idx = 0
    iterate_list = iterate_list[:]

    # ============ initialise lmdb key ============
    idx = 0

    for each in tqdm(iterate_list, desc="generating lmdb for " + prefix , leave=True):
        img_id = each[0]
        info = dict(each[1])
        # img_id = u'201462516952791918769'
        joints = info["joints"][1:]
        attrs = info['attribute']
        bboxs = info["bbox"]
        print "hahaha"
        print attrs
        print json.dumps(info, indent=4)
        print "hahaha"
        img = cv.imread(cfg.data_path + "img/" + img_id + ".jpg")
        bbox_json[img_id] = {}
        bbox_json[img_id]["whole"] = bboxs
        bbox_json[img_id]["gt_bbox"] = []
        for temp_img, temp_joints in data_augment(img, joints, bboxs, flip=cfg.augment_flip):
            # === plot image ===
            # temp_img_for_show = np.array(temp_img, copy=True)
            #
            # for pt_idx, pt in enumerate(temp_joints[:]):
            #     temp_color = (255,0,0)
            #     if 6 <= pt_idx:
            #         temp_color = (0, 255, 0)
            #     cv.circle(temp_img_for_show, center=(pt[0], pt[1]), radius=1, color=temp_color, thickness=10)
            # cv.imshow("show", temp_img_for_show)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # ============ update lmdb key ============
            lmdb_key = str(idx).encode('utf-8')

            # ============ generate image lmdb ============
            img_datum = lu.generate_img_datum(temp_img)
            img_txn.put(lmdb_key, img_datum.SerializeToString())

            # ============ generate joints lmdb ============

            joint_datum = np.asarray(temp_joints[1:]) / 227.0
            joint_datum = joint_datum.ravel()
            joint_datum = lu.generate_array_datum(joint_datum, is_float=True)
            # print("joints info:")
            # print(joint_datum)

            joint_txn.put(lmdb_key, joint_datum.SerializeToString())

            # ============ generate attribute lmdb ============
            def find_idx(d):
                for th, item in enumerate(list(d)):
                    if d[item]==True:
                        return th
                return -1

            for kth, each in enumerate(attribute_name):
                temp = attrs[each]
                res = find_idx(temp)

                # print temp, res, each
                res_array = np.zeros((1, 1)).ravel().astype(int)
                res_array[0] = res
                # print res_array,
                res_datum = lu.generate_array_datum(res_array, is_float=False)
                attribute_txn[kth].put(lmdb_key, res_datum.SerializeToString())


            # ============ generate bbox lmdb ============
            # temp_joints = info["joints"]
            # print temp_joints
            # print info["joints"]
            # exit(0)
            bbox_data = []

            # toursor
            toursor_pts = [temp_joints[0], temp_joints[3], temp_joints[8]] # no longer left hand + right hand + spine
            temp_pts = toursor_pts
            res = find_sub_bbox(temp_pts, (227, 227),0.85, 0.88) / float(227)
            bbox_data.append(res)

            # collar
            collar_pts = [temp_joints[0], temp_joints[3], temp_joints[6]] # head + left shoulder + right shoulder + spine 1
            temp_pts = collar_pts
            res = find_sub_bbox(temp_pts, (227, 227),0.85, 1.15) / float(227)
            bbox_data.append(res)

            # button_placket
            button_placket_pts = [temp_joints[0], temp_joints[3], temp_joints[8]] # Left hand 1 + right hand 3 + spine 2
            temp_pts = button_placket_pts
            res = find_sub_bbox(temp_pts, (227, 227),0.85, 0.9) / float(227)
            bbox_data.append(res)

            # =================left_sleeve=================
            left_sleeve = [temp_joints[0], temp_joints[1], temp_joints[2]] # Left hand 1 2 3
            temp_pts = left_sleeve

            legal = True
            for each in left_sleeve:
                if each[0] < 1 or each[1] < 1:
                    legal = False
                    break
            if legal == False:
                temp_pts = temp_joints[1:]
            res = find_sub_bbox(temp_pts, (227, 227),0.8, 1.02, limit_ratio=0.6) / float(227)
            bbox_data.append(res)


            # =================right_sleeve=================
            right_sleeve = [temp_joints[3], temp_joints[4], temp_joints[5]] # Right hand 1 2 3
            temp_pts = right_sleeve

            legal = True
            for each in left_sleeve:
                if each[0] < 1 or each[1] < 1:
                    legal = False
                    break
            if legal == False:
                temp_pts = temp_joints[1:]
            res = find_sub_bbox(temp_pts, (227, 227),0.8, 1.02, limit_ratio=0.6) / float(227)
            bbox_data.append(res)

            if True:
                test_img = np.array(temp_img, copy=True)
                res = res * float(227)
                cv.rectangle(test_img, (int(res[0]), int(res[1])), (int(res[2]), int(res[3])), (225, 0, 0), 2)
                pt_list = temp_joints
                for pt in pt_list:
                    cv.circle(test_img, (int(pt[0]), int(pt[1])), 1, (0, 225, 0), 2)
                for pt in temp_pts:
                    cv.circle(test_img, (int(pt[0]), int(pt[1])), 1, (0, 0, 225), 2)
                cv.imshow("image", test_img)
                cv.waitKey(0)
                cv.destroyAllWindows()

            # bbox_json[key] = list(bbox_data)
            bbox_json[img_id]["gt_bbox"].append([each.tolist() for each in bbox_data])
            # print list(bbox_data[0, :])
            # print(bbox_data)

            for ith in range(len(bbox_data)):
                # print(idx)
                # res_datum = lu.generate_array_datum(bbox_data[ith], is_float=False)
                bbox_temp_joints = np.asarray(bbox_data[ith]).ravel()
                res_datum = lu.generate_array_datum(bbox_temp_joints, is_float=True)
                bbox_txn[ith].put(lmdb_key, res_datum.SerializeToString())
            #     print res_datum
            # print("=====================================================")
            if False:
                test_img = np.array(temp_img, copy=True)
                res = res * float(227)
                cv.rectangle(test_img, (int(res[0]),int(res[1])), (int(res[2]),int(res[3])), (225, 0, 0), 2)
                pt_list = temp_joints
                for pt in pt_list:
                    cv.circle(test_img, (int(pt[0]), int(pt[1])), 1,  (0, 225, 0), 2)
                for pt in temp_pts:
                    cv.circle(test_img, (int(pt[0]), int(pt[1])), 1,  (0, 0, 225), 2)
                cv.imshow("image", test_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
            # ============ update lmdb key ============
            idx += 1

            # ============ Don't augment for valid =======
            if prefix == "valid":
                break

    with open(folder + 'lmdb/' + prefix + "_bbox.json", 'w') as fp:
        json.dump(bbox_json, fp, indent=4)

    img_txn.commit()
    img_env.close()

    joint_txn.commit()
    joint_env.close()

    [each.commit() for each in attribute_txn]
    [each.close() for each in attribute_env]

    [each.commit() for each in bbox_txn]
    [each.close() for each in bbox_env]
    pass


if __name__ == "__main__":

    # cv.namedWindow('origin')

    with open(cfg.data_path + "augmented_data.json", 'r') as fp:
        data = json.load(fp)
    L = len(data.items()) * 4 / 5
    iterate_list = data.items()
    # random.shuffle(iterate_list)
    generate(iterate_list[:L],cfg.data_path, prefix="train")
    generate(iterate_list[L:],cfg.data_path, prefix="valid")

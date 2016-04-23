import cv2
import numpy as np
import glob, sys, os, shutil
import json
from os.path import join as pjoin
from read_tool import *

useful_part = ['head', 'left_hand', 'right_hand', 'spine']

DEBUG_MODE = True


def joint_filter(image_id, image_joint):
    new_joint = {}
    for each in image_id[:]:
        # print image_joint[each]['head']
        res = []
        # If 'head' is invisible, skip this image
        if image_joint[each]['head'][0]['visible'] == False:
            image_id.remove(each)
            del image_joint[each]
            continue

        for part in useful_part:
            for pt in image_joint[each][part]:
                res.append(pt['position'])

        negative_position = False
        for pos in res:
            if pos[0] < 0.5 or pos[1] < 0.5: #-1
                negative_position = True
                break

        if negative_position:
            image_id.remove(each)
            del image_joint[each]
            continue
        else:
            new_joint[each] = res

    return image_id, new_joint


def attribute_filter(image_id, image_attr, image_type):
    new_attr = {}
    new_type = {}

    key_list = ['Placket1', 'SleeveLength', 'CollarType', 'ButtonType', 'Cloth_Type']
    new_list = ['placket', 'sleeve_type', 'collar_type', 'button_type', 'cloth_type']


    for img_id in image_id[:]:
        if img_id not in image_attr or img_id not in image_type:
            image_id.remove(img_id)
            continue

        temp = {new_list[key_list.index(item)]: image_attr[img_id][item] for item in image_attr[img_id] if
                item in key_list}
        temp['cloth_type'] = image_type[img_id]['Cloth_Type']

        new_attr[img_id] = temp

    return combine_attribute(image_id, new_attr)


def combine_attribute(image_id, image_attr):
    count_table = {}
    for img_id in image_id:
        for attr in image_attr[img_id]:
            temp = image_attr[img_id]
            if attr not in count_table:
                count_table[attr] = {}
            if temp[attr] not in count_table[attr]:
                count_table[attr][temp[attr]] = 0
            count_table[attr][temp[attr]] += 1

    # Todo
    # Select
    # print len(count_table)
    select_3_largest = lambda x: dict(sorted(x.items(), key=lambda x: x[1], reverse=True)[:3])
    select_4_largest = lambda x: dict(sorted(x.items(), key=lambda x: x[1], reverse=True)[:4])
    count_table = {each: select_3_largest(count_table[each]) \
        if each != "cloth_type" else count_table[each] \
                   for each in count_table}
    count_table = {each: select_4_largest(count_table[each]) \
        if each == "cloth_type" else count_table[each] \
                   for each in count_table}

    for img_id in image_id[:]:
        cur = {}
        for attr in image_attr[img_id]:
            temp = image_attr[img_id]
            # print temp[attr]
            cur[attr] = {each: False for each in count_table[attr]}
            cur[attr]["else"] = False

            if temp[attr] in count_table[attr]:
                cur[attr][temp[attr]] = True
            else:
                cur[attr]["else"] = True
            # print cur[attr]
        image_attr[img_id] = cur
    # print image_attr[img_id]
    return image_id, image_attr


def data_filter(image_id, image_joint, image_attr, image_type):
    image_id, image_joint = joint_filter(image_id, image_joint)
    image_id, image_attr = attribute_filter(image_id, image_attr, image_type)

    return image_id, image_joint, image_attr

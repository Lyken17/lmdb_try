import cv2
import numpy as np
import glob, sys, os, shutil
import json
from os.path import join as pjoin

HOME = "/home/cs-user"
DROPBOX = pjoin(HOME, 'Dropbox', 'Data' )

def read_in_img(img_folder_path):
    return list_files(img_folder_path, "jpg")

def read_in_joint(joint_folder_path):
    joint_list = list_files(joint_folder_path, "json")
    image_joint = {}
    for each in joint_list[:]:
        with open(pjoin(joint_folder_path, each + ".json")) as fp:
            joint = json.load(fp)
        image_joint[each] = joint
    return image_joint


def list_files(path, ext):
    L =  glob.glob1(path, "*." + ext)
    return  [each.split('.')[0] for each in L]


def read_in_type(type_path):
    pass

def read_info():
    image_arr = read_in_img('Data/img')
    image_joint = read_in_joint("Data/joint")
    with open('Data/attribute.json') as fp:
       image_attr = json.load(fp)
    with open('Data/type.json') as fp:
        image_type = json.load(fp)

    return [image_arr, image_joint, image_attr]

def extract_useful():
    img_folders = [pjoin(DROPBOX, 'img'), \
                   pjoin(DROPBOX, 'labeled_img')]
    label_folders = [pjoin(DROPBOX, 'current'), \
                     pjoin(DROPBOX, 'new_label')]

    aim_img_folder = pjoin(HOME, "PycharmProjects/beihang_lmdb/Data/img/")
    aim_jnt_folder = pjoin(HOME, "PycharmProjects/beihang_lmdb/Data/joint/")

    for idx, item in enumerate(img_folders[:]):
        img_folder_path = img_folders[idx]
        label_folder_path = label_folders[idx]

        image_arr = list_files(img_folder_path, "jpg")
        label_arr = list_files(label_folder_path, "json")

        image_arr = label_arr = list(set(image_arr) & set(label_arr))

        print len(image_arr)
        for img_id in image_arr:
            src_dir = pjoin(img_folder_path, img_id + ".jpg")
            aim_dir = pjoin(aim_img_folder, img_id + ".jpg")
            shutil.copyfile(src_dir, aim_dir)

            src_dir = pjoin(label_folder_path, img_id + ".json")
            aim_dir = pjoin(aim_jnt_folder, img_id + ".json")
            shutil.copyfile(src_dir, aim_dir)
    pass


if __name__ == "__main__":
    read_info()
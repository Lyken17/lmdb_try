import cv2
import numpy as np
import glob, sys, os, shutil
import json
from os.path import join as pjoin
import random, copy
from data_process import *
from image_process import *

HOME = "/Users/lykensyu"
DROPBOX = pjoin(HOME, 'Dropbox', 'Data_old')

data_source = "Local"
if data_source.lower() == "DROPBOX".lower():
    arr_path = pjoin(DROPBOX, 'img/')
    joint_path = pjoin(DROPBOX, 'joint/')
else:
    arr_path = 'Data/img/'
    joint_path = 'Data/joint/'


def read_in_img(img_folder_path):
    return list_files(img_folder_path, "jpg")


def read_in_joint(joint_folder_path):
    joint_list = list_files(joint_folder_path, "json")
    image_joint = {}
    for each in joint_list[:]:
        with open(pjoin(joint_folder_path, each + ".json")) as fp:
            joint = json.load(fp)
        image_joint[str(each)] = joint
    return image_joint


def list_files(path, ext):
    L = glob.glob1(path, "*." + ext)
    return [each.split('.')[0] for each in L]


def read_in_type(type_path):
    pass


def read_info(option=0):
    if option == 0:
        # Read in all necessary data
        image_id = read_in_img(arr_path)
        image_joint = read_in_joint(joint_path)
        # print "%d images" % len(image_id)
        with open('Data/attribute.json', 'r') as fp:
            image_attr = json.load(fp)
            # print "%d attributions" % len(image_attr)
        with open('Data/type.json', 'r') as fp:
            image_type = json.load(fp)
            # print "%d cloth types" % len(image_type)

        '''
        image_attr = {each["ID_image"]: each for each in image_attr}
        image_type = {each["ID_Image"]: each for each in image_type}
        with open('Data/attribute_1.json', 'w') as fp:
            json.dump(image_attr, fp, indent=4)
        with open('Data/type_1.json', 'w') as fp:
            json.dump(image_type, fp, indent=4)
        '''
        # Preprocess the data
        image_id, image_joint, image_attr = data_filter(image_id, image_joint, image_attr, image_type)

        f1 = open('Data/' + 'image_id.json', 'w')
        f2 = open('Data/' + 'image_joint.json', 'w')
        f3 = open('Data/' + 'image_attr.json', 'w')

        json.dump(image_id, f1, indent=4)
        json.dump(image_joint, f2, indent=4)
        json.dump(image_attr, f3, indent=4)

        f1.close()
        f2.close()
        f3.close()
    else:
        f1 = open('Data/' + 'image_id.json', 'r')
        f2 = open('Data/' + 'image_joint.json', 'r')
        f3 = open('Data/' + 'image_attr.json', 'r')

        image_id = json.load(f1)
        image_joint = json.load(f2)
        image_attr = json.load(f3)

        f1.close()
        f2.close()
        f3.close()

    return [image_id, image_joint, image_attr]


def data_augmentation(image_id, image_joint, image_attr,prefix="train"):
    random.shuffle(image_id)
    augment_times = 4
    output_dir = "Data/output"
    prefix = "valid"
    img_dir = pjoin(output_dir, prefix + "_img/")

    final_res = {}
    image_id_list = image_id[:2710] if prefix == "train" else image_id[2711:]
    for img_id in image_id_list:
    # for img_id in [str(2014628214953250182141)]:
        # print img_id
        # image_joint[img_id]

        count = 0
        for i in xrange(augment_times):
            img = cv2.imread(pjoin(arr_path, img_id + ".jpg"))
            for new_img, new_pts in (augment_data(img, image_joint[img_id], flip=True)):
                temp_id = img_id + "_augment_" + str(count)
                count += 1
                final_res[temp_id] = {}
                final_res[temp_id]["joints"] = new_pts
                final_res[temp_id]["attribute"] = image_attr[img_id]
                cv2.imwrite(pjoin(img_dir, temp_id + ".jpg"), new_img)

    print len(final_res)
    with open(pjoin(output_dir, prefix + "_data.json"), 'w') as fp:
        json.dump(final_res, fp, indent=4)


def extract_useful():
    img_folders = [pjoin(DROPBOX, 'img'),
                   pjoin(DROPBOX, 'labeled_img')]
    label_folders = [pjoin(DROPBOX, 'current'),
                     pjoin(DROPBOX, 'new_label')]

    project_path = os.getcwd()
    aim_img_folder = pjoin(project_path, "Data/img/")
    aim_jnt_folder = pjoin(project_path, "Data/joint/")

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
    # extract_useful()
    image_id, image_joint, image_attr = read_info(option=1)
    data_augmentation(image_id, image_joint, image_attr)

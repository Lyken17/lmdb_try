import cv2
import numpy as np
import glob, sys, os, shutil
import json
from os.path import join as pjoin
from read_tool import *
from data_process import *


class lmdb_img():
    def __init__(self, img_id, joints, path=None):
        self.id = img_id
        self.path = path
        self.joints = joints
        self.image = cv2.imread(pjoin(path, img_id + ".jpg"))

    def generate_bbox(self):
        h, w = self.image.shape[:2]
        joint = np.array(self.joints)
        print joint
        X_list = joint[:, 0]
        Y_list = joint[:, 1]
        X_list = [each for each in X_list if each != -1]
        Y_list = [each for each in Y_list if each != -1]

        # Todo
        # -1 correction
        xx_min = x_min = min(X_list)
        xx_max = x_max = max(X_list)
        yy_min = y_min = min(Y_list)
        yy_max = y_max = max(Y_list)

        # print x_min, y_min, x_max, y_max
        mu, sigma = 0, 0.3
        width = max(x_max - x_min, y_max - y_min)

        print "image shape %d %d" % self.image.shape[:2]
        print x_min, y_min, x_max, y_max
        x_min = int(min(max(1, x_min - width * abs(np.random.normal(mu, sigma))), max(1, x_min - 10)))
        y_min = int(min(max(1, y_min - width * abs(np.random.normal(mu, sigma))), max(1, y_min - 10)))
        x_max = int(max(min(h, x_max + width * abs(np.random.normal(mu, sigma))), min(h, x_max + 10)))
        y_max = int(max(min(w, y_max + width * abs(np.random.normal(mu, sigma))), min(w, y_max + 10)))

        height, weight = y_max - y_min, x_max - x_min
        ratio = height / float(weight)

        print "bbox : ",
        print x_min, y_min, x_max, y_max
        # print "origin ratio : %f" % ratio
        # print height, weight

        if 0.8 <= ratio <= 1.25:
            pass
        elif ratio > 1.25:
            print "ratio < 0.8"
            x_min = int(max(0, x_min - weight * 0.1))
            x_max = int(min(w, x_max + weight * 0.1))
        elif ratio < 0.8:
            print "ratio > 1.25"
            y_min = int(max(0, y_min - height * 0.1))
            y_max = int(min(h, y_max + height * 0.1))

        x_min = min(x_min, xx_min)
        y_min = min(y_min, yy_min)
        x_max = max(x_max, xx_max)
        y_max = max(y_max, yy_max)

        height, weight = y_max - y_min, x_max - x_min
        ratio = height / float(weight)
        # print "after ratio : %f " % ratio
        # print height, weight
        print "after bbox : ",
        print x_min, y_min, x_max, y_max

        temp_img = np.array(self.image, copy=True)
        for pt in self.joints:
            cv2.circle(temp_img, tuple(pt), 1, (128, 128, 0), 15)
        temp_img = temp_img[y_min:y_max, x_min:x_max]

        # cv2.imwrite("temp/" + self.id + str(random.randint(0, 100)) + ".jpg", temp_img)

        cv2.imshow("try", temp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return [x_min, y_min], [x_max, y_max]

    def crop(self, bbox):
        pass

    def flip(self):
        pass

    def augment(self):
        self.generate_bbox()

    def show(self, option=0):
        if option == 0:
            img = np.array(self.image, copy=True)
        else:
            img = np.array(self.image, copy=True)
            # print self.joints
            for pt in self.joints:
                # print pt
                cv2.circle(img, tuple(pt), 1, (128, 128, 0), 15)

        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img


def show(image, joints=None, option=0):
    if option == 0:
        img = np.array(image, copy=True)
    else:
        img = np.array(image, copy=True)
        # print self.joints
        for pt in joints:
            # print pt
            cv2.circle(img, tuple(pt), 1, (128, 128, 0), 10)
            pass
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


def augment_data(img, pts, flip=True, imshow=False):
    pt1, pt2 = generate_bbox(img, pts)
    new_img, new_pts = crop_and_resize(img, pt1, pt2, pts)
    if imshow:
        show(new_img, new_pts, option=1)
    yield new_img, new_pts

    if flip == True:
        new_img = cv2.flip(new_img, 1)
        new_pts = [[227 - pt[0], pt[1]] for pt in new_pts]
        if imshow:
            show(new_img, new_pts, option=1)
        yield new_img, new_pts


def crop_and_resize(img, pt1, pt2, pts):
    x_min = pt1[0]
    y_min = pt1[1]
    x_max = pt2[0]
    y_max = pt2[1]

    h_scale = 227 / float(x_max - x_min)
    w_scale = 227 / float(y_max - y_min)

    new_img = np.array(img[y_min:y_max, x_min:x_max], copy=True)
    new_pts = [[pt[0] - x_min, pt[1] - y_min] for pt in pts]

    new_img = cv2.resize(new_img, (227, 227))
    new_pts = [[int(pt[0] * h_scale), int(pt[1] * w_scale)] for pt in new_pts]

    return new_img, new_pts


def generate_bbox(img, pts):
    h, w = img.shape[:2]
    joint = np.array(pts)

    # print joint

    X_list = joint[:, 0]
    Y_list = joint[:, 1]
    X_list = [each for each in X_list if each > 0]
    Y_list = [each for each in Y_list if each > 0]

    xx_min = x_min = min(X_list)
    xx_max = x_max = max(X_list)
    yy_min = y_min = min(Y_list)
    yy_max = y_max = max(Y_list)

    # print x_min, y_min, x_max, y_max
    mu, sigma = 0, 0.3
    width = max(x_max - x_min, y_max - y_min)

    # print "image shape %d %d" % img.shape[:2]
    # print x_min, y_min, x_max, y_max
    x_min = int(min(max(1, x_min - width * abs(np.random.normal(mu, sigma))), max(1, x_min - 10)))
    y_min = int(min(max(1, y_min - width * abs(np.random.normal(mu, sigma))), max(1, y_min - 10)))
    x_max = int(max(min(h, x_max + width * abs(np.random.normal(mu, sigma))), min(h, x_max + 10)))
    y_max = int(max(min(w, y_max + width * abs(np.random.normal(mu, sigma))), min(w, y_max + 10)))

    height, weight = y_max - y_min, x_max - x_min
    ratio = height / float(weight)

    if 0.8 <= ratio <= 1.25:
        pass
    elif ratio > 1.25:
        # print "ratio < 0.8"
        x_min = int(max(0, x_min - weight * 0.1))
        x_max = int(min(w, x_max + weight * 0.1))
    elif ratio < 0.8:
        # print "ratio > 1.25"
        y_min = int(max(0, y_min - height * 0.1))
        y_max = int(min(h, y_max + height * 0.1))

    x_min = min(x_min, xx_min)
    y_min = min(y_min, yy_min)
    x_max = max(x_max, xx_max)
    y_max = max(y_max, yy_max)

    height, weight = y_max - y_min, x_max - x_min
    ratio = height / float(weight)

    '''
    temp_img = np.array(img, copy=True)
    for pt in pts:
        cv2.circle(temp_img, tuple(pt), 1, (128, 128, 0), 15)
    temp_img = temp_img[y_min:y_max, x_min:x_max]

    cv2.imshow("try", temp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    return [x_min, y_min], [x_max, y_max]


def test_img():
    with open("Data/output/train_data.json", 'r') as fp:
        data = json.load(fp)
    for img_id in data:
        img = cv2.imread("Data/output/train_img/" + img_id + ".jpg")
        # print "Data/output/train_img" + img_id + ".jpg"

        pts = data[img_id]["joints"]
        # print pts
        show(img, joints=pts, option=1)



if __name__ == "__main__":
    test_img()
    pass
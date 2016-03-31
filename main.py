import cv2
import numpy as np
import glob, sys, os, shutil
import json
from os.path import join as pjoin
import random, copy
from tqdm import tqdm

from data_process import *
from image_process import *
from read_tool import *

# extract_useful()
image_id, image_joint, image_attr = read_info(option=read_from_file)
print len(image_id)
data_augmentation(image_id, image_joint, image_attr)
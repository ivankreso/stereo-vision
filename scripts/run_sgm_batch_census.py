#!/usr/bin/python
import os
import subprocess
from os.path import isfile, join

#data_folder = "/home/kivan/datasets/KITTI/stereo/data_stereo_flow/training/"
data_folder = "/home/kivan/datasets/KITTI/stereo/data_stereo_flow/testing/"
left_dir = data_folder + "/image_0/"
right_dir = data_folder + "/image_1/"

binary_path = "/home/kivan/source/cv-stereo/build/sgm_single/release/sgm_single"
out_folder = "/home/kivan/source/deep-metric-learning/output/sgm_census/"

#binary_path = "/home/kivan/source/cv-stereo/build/spsstereo/release/sgm_stereo"
#out_folder = "/home/kivan/source/deep-metric-learning/output/sgm_yama/"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
    os.makedirs(out_folder + "/disparities/")
    os.makedirs(out_folder + "/norm_hist/")
    os.makedirs(out_folder + "/interpolation/")
else:
    print("WARNING: path exists - ", out_folder)

filelist = [f for f in os.listdir(left_dir) if isfile(join(left_dir,f)) and "_10.png" in f]

for filename in filelist:
    print(filename)
    #prefix = filename[:9]
    #num_str = "%06d" % (i)
    img_left = left_dir + filename
    img_right = right_dir + filename
    #subprocess.call([binary_path, img_left, img_right, out_folder, "5", "80"])
    subprocess.call([binary_path, img_left, img_right, out_folder, "3", "60"])

    #subprocess.call([binary_path, img_left, img_right, out_folder])

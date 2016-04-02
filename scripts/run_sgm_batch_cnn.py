#!/usr/bin/python
import os
import subprocess
from os.path import isfile, join

binary_path = "/home/kivan/source/cv-stereo/build/sgm_cnn/release/sgm_cnn"
#binary_path = "/home/kivan/source/cv-stereo/build/sgm_cnn_new/release/sgm"

#data_folder = "/home/kivan/source/deep-metric-learning/output/results/ThuJun1122:16:492015/results/"
#out_folder = "/home/kivan/source/deep-metric-learning/output/results/ThuJun1122:16:492015/results/depth/"
#data_dir = "/home/kivan/source/deep-metric-learning/output/results/ThuJun1800:22:462015/results/"
#data_dir = "/home/kivan/source/deep-metric-learning/output/results/ThuJun1822:33:402015/results/"
#data_dir = '/home/kivan/source/deep-metric-learning/output/results/FriJun1900:57:092015/results/'
#data_dir = '/home/kivan/source/deep-metric-learning/output/results/ThuSep316:10:232015/results/test/'
#data_dir = '/home/kivan/source/deep-metric-learning/output/learned_models/WedSep913:32:102015/results/test/'
#data_dir = '/home/kivan/source/deep-metric-learning/output/learned_models/FriSep1100:25:162015/results/train/'
#data_dir = '/home/kivan/source/deep-metric-learning/output/learned_models/Fri11Sep201503:21:51PMCEST/results/train'
#data_dir = '/home/kivan/source/deep-metric-learning/output/results/Wed23Sep201510:35:12PMCEST/'
#data_dir = '/home/kivan/source/deep-metric-learning/output/learned_models/Wed23Sep201510:35:12PMCEST/results/train/'
data_dir = '/home/kivan/source/deep-metric-learning/output/learned_models/ThuSep2414:54:062015/results/train/'

data_folder = data_dir + "/representation/"
out_folder = data_dir + "/depth/"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
    os.makedirs(out_folder + "/disparities/")
    os.makedirs(out_folder + "/norm_hist/")
    os.makedirs(out_folder + "/interpolation/")
else:
    print("WARNING: path exists - ", out_folder)

filelist = [f for f in os.listdir(data_folder) if isfile(join(data_folder,f)) and "left" in f]

for filename in filelist:
    print(filename)
    prefix = filename[:9]
    #num_str = "%06d" % (i)
    #num_str = "%010d" % (i)
    img_left = data_folder + filename
    img_right = data_folder + prefix + "_right.bin"
    subprocess.call([binary_path, img_left, img_right, out_folder, prefix, "3", "60"])
    #subprocess.call([binary_path, img_left, img_right, out_folder, prefix, "1", "32"])
    #subprocess.call([binary_path, img_left, img_right, out_folder, "3", "60", "1"])

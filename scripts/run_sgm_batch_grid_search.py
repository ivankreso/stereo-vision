#!/usr/bin/python

# KITTI 02
#start_num = 0
#stop_num = 4660
#frame_step = 1
#left_prefix = "/image_0/"
#right_prefix = "/image_1/"
#left_suffix = ".png"
#right_suffix = ".png"
#out_fname = "kitti_02_lst.xml"

import os
import subprocess

start_num = 0
stop_num = 193
#stop_num = 1100
frame_step = 1
left_suffix = "_10.png"
right_suffix = "_10.png"

#data_folder = "/home/kivan/Projects/datasets/KITTI/sequences_gray/07/"
data_folder = "/home/kivan/Projects/datasets/KITTI/dense_stereo/training/"

#left_prefix = "/colored_0/"
#right_prefix = "/colored_1/"
#out_folder_prefix = "/home/kivan/Projects/datasets/results/dense_stereo/spsstereo/kitti/experiment"
#binary_path = "/home/kivan/Projects/cv-stereo/build/spsstereo/release/spsstereo"

left_prefix = "/image_0/"
right_prefix = "/image_1/"
out_folder_prefix = "/home/kivan/Projects/datasets/results/dense_stereo/kitti/experiment"
binary_path = "/home/kivan/Projects/cv-stereo/build/our_sgm/release/our_sgm"

#P1 = [3, 5, 7, 10, 13]
#P2 = [30, 40, 50, 70, 90]
P1 = [4, 5, 6]
P2 = [40, 45, 50, 55, 60]
for penalty1 in P1:
    for penalty2 in P2:
        out_folder = out_folder_prefix + "_" + str(penalty1) + "_" + str(penalty2) + "/data/"
        print(out_folder)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        else:
            print("WARNING: path exists - ", out_folder)
        for i in range(start_num, stop_num+1, frame_step):
            num_str = "%06d" % (i)
            #num_str = "%010d" % (i)
            img_left = data_folder + left_prefix + num_str + left_suffix
            img_right = data_folder + right_prefix + num_str + right_suffix
            print(img_left)
            subprocess.call([binary_path, img_left, img_right, out_folder, str(penalty1), str(penalty2)])
            #cmd = binary_path + " " + img_left + " " + img_right + " " + out_folder
            #subprocess.call([cmd], shell=True)

#ofile.write("</imagelist>\n</opencv_storage>")
#ofile.close()


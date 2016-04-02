#!/usr/bin/python
import os
import subprocess
from os.path import isfile, join

# KITTI 02
#start_num = 0
#stop_num = 4660
#frame_step = 1
#left_prefix = "/image_0/"
#right_prefix = "/image_1/"
#left_suffix = ".png"
#right_suffix = ".png"
#out_fname = "kitti_02_lst.xml"

#start_num = 0
##stop_num = 1100
#frame_step = 1
#left_suffix = "_10.png"
#right_suffix = "_10.png"
#
##data_folder = "/home/kivan/Projects/datasets/KITTI/sequences_gray/07/"
##data_folder = "/home/kivan/Projects/datasets/KITTI/dense_stereo/training/"
##stop_num = 193
#data_folder = "/home/kivan/Projects/datasets/KITTI/dense_stereo/testing/"
#stop_num = 194
#
##left_prefix = "/colored_0/"
##right_prefix = "/colored_1/"
##out_folder = "/home/kivan/Projects/datasets/results/dense_stereo/spsstereo/kitti/data"
##binary_path = "/home/kivan/Projects/cv-stereo/build/spsstereo/release/spsstereo"
#
#left_prefix = "/image_0/"
#right_prefix = "/image_1/"
##out_folder = "/home/kivan/Projects/datasets/results/dense_stereo/kitti/testing/our_sgm_5_60/data/"
##binary_path = "/home/kivan/Projects/cv-stereo/build/our_sgm/release/our_sgm"

binary_path = "/home/kivan/source/cv-stereo/build/spsstereo/release/spsstereo"
#binary_path = "/home/kivan/source/cv-stereo/build/sgm_single/release/sgm_single"

#data_folder = "/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/Training_00/RGB/"
#out_folder = "/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/Training_00/RGB/depth"
#img_right_dir = "/home/kivan/datasets/KITTI/sequences_color/00/image_3/"

#data_folder = "/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/Validation_07/RGB/"
#out_folder = "/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/Validation_07/RGB/depth"
#img_right_dir = "/home/kivan/datasets/KITTI/sequences_color/07/image_3/"

#data_folder = "/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/sunando_sengupta/train/images/"
#out_folder = "/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/sunando_sengupta/train/depth/"
data_folder = "/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/sunando_sengupta/test/images/"
out_folder = "/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/sunando_sengupta/test/depth/"
img_right_dir = "/home/kivan/datasets/KITTI/sequences_color/"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
else:
    print("WARNING: path exists - ", out_folder)

left_dir = data_folder + "/left/"
right_dir = data_folder + "/right/"
filelist = [f for f in os.listdir(left_dir) if isfile(join(left_dir,f))]

for filename in filelist:
    print(filename)
    #num_str = "%06d" % (i)
    #num_str = "%010d" % (i)
    img_left = left_dir + filename
    img_right = right_dir + filename
    right_src_img = img_right_dir + filename[0:2] + '/image_3/' + filename[3:]
    print(right_src_img)
    subprocess.call(["/bin/cp", right_src_img, img_right])
    subprocess.call([binary_path, img_left, img_right, out_folder])
    #subprocess.call([binary_path, img_left, img_right, out_folder, "5", "60"])

    #cmd = binary_path + " " + img_left + " " + img_right + " " + out_folder
    #subprocess.call([cmd], shell=True)

#ofile.write("</imagelist>\n</opencv_storage>")
#ofile.close()

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

# KITTI 19
#start_num = 0
#stop_num = 1200
#frame_step = 1
#left_prefix = "/image_0/"
#right_prefix = "/image_1/"
#left_suffix = ".png"
#right_suffix = ".png"
#out_fname = "kitti_10_lst.xml"

# Axel tractor dataset
start_num = 1
stop_num = 223
frame_step = 1
left_prefix = "/cam1_"
right_prefix = "/cam2_"
left_suffix = ".pgm"
right_suffix = ".pgm"
out_fname = "tractor.xml"

#start_num = 1
#stop_num = 1800
#frame_step = 1
#left_prefix = "left/tsukuba_fluorescent_L_"
#right_prefix = "right/tsukuba_fluorescent_R_"
#left_suffix = ".png"
#right_suffix = ".png"
#out_fname = "tsukuba_fluorescent_lst.xml"

ofile = open(out_fname, 'w')
ofile.write("<?xml version=\"1.0\"?>\n<opencv_storage>\n<imagelist>\n")

for i in range(start_num, stop_num+1, frame_step):
   #num_str = "%06d" % (i)
   num_str = "%010d" % (i)
   img_left = "\"" + left_prefix + num_str + left_suffix + "\"\n"
   img_right = "\"" + right_prefix + num_str + right_suffix + "\"\n"
   ofile.write(img_left)
   ofile.write(img_right)

ofile.write("</imagelist>\n</opencv_storage>")
ofile.close()


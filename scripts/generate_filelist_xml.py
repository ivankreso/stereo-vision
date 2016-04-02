#!/usr/bin/python

#file_start = 1
## viso00
##file_end = 4540
## 170m sim path
#file_end = 186
#file_step = 1
#prefix = "point_projs_"
#suffix = ".txt"
#out_fname = "stereosim_lst.xml"

# KITTI 02
file_start = 0
file_end = 4660
file_step = 1
prefix = "disp_"
suffix = ".png"
out_fname = "kitti_disp_02_lst.xml"

ofile = open(out_fname, 'w')
ofile.write("<?xml version=\"1.0\"?>\n<opencv_storage>\n<imagelist>\n")

for i in range(file_start, file_end+1, file_step):
   num_str = "%06d" % (i)
   filename = "\"" + prefix + num_str + suffix + "\"\n"
   ofile.write(filename)

ofile.write("</imagelist>\n</opencv_storage>")
ofile.close()

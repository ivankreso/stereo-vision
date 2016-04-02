#!/usr/bin/python
import os, sys, re
import subprocess

if len(sys.argv) != 3:
    print("Usage:\n\t\t" + sys.argv[0] + " src_dir/ dst_dir/\n")
    sys.exit(1)

# create output dir
if not os.path.exists(sys.argv[2]):
    os.makedirs(sys.argv[2])

# get file list of input dir
imglst = os.listdir(sys.argv[1])

# filter only pgm images
regex = re.compile(".*\.raw$", re.IGNORECASE)
imglst = [f for f in imglst if regex.search(f)]
imglst.sort()

# split images
for i in range(len(imglst)):
    print(str(i/len(imglst)*100.0)[:5] + "%\t" + imglst[i])
    #os.system("./stereo_pgm_to_png " + sys.argv[1] + imglst[i] + " " + sys.argv[2] + imglst[i][:-4])
    subprocess.call(["./stereo_raw_to_png", sys.argv[1] + imglst[i], sys.argv[2] + imglst[i][:-4]])


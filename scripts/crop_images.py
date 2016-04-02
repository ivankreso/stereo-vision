#!/usr/bin/python

# Note: python3 script
import os, sys, re

if len(sys.argv) != 3:
   print("Usage:\n\t\t" + sys.argv[0] + " src_dir/ dst_dir/\n")
   sys.exit(1)

# create output dir
if not os.path.exists(sys.argv[2]):
   os.makedirs(sys.argv[2])

# get file list of input dir
imglst = os.listdir(sys.argv[1])

# filter only appropriate images
#regex = re.compile(".*\.png$", re.IGNORECASE)          
regex = re.compile(".*\.pgm$", re.IGNORECASE)          
imglst = [f for f in imglst if regex.search(f)]
imglst.sort()

# split images
for i in range(len(imglst)):
   print(str(i/len(imglst)*100.0)[:5] + "%\t" + imglst[i])
   #os.system("convert -crop 590x362+23+35 " + sys.argv[1] + imglst[i] + " " + sys.argv[2] + imglst[i])
   # +repage to remove offset information after cropping that some formats like png and gif stores

   # tractor dataset
   #os.system("convert -crop 1183x934+49+40 +repage " + sys.argv[1] + imglst[i] + " " + sys.argv[2] + imglst[i])   
   os.system("convert -crop 1183x810+49+40 +repage " + sys.argv[1] + imglst[i] + " " + sys.argv[2] + imglst[i])   

   #os.system("convert -crop 640x426+0+0 +repage " + sys.argv[1] + imglst[i] + " " + sys.argv[2] + imglst[i])   
   #os.system("convert -crop 590x362+23+35 +repage " + sys.argv[1] + imglst[i] + " " + sys.argv[2] + imglst[i])
   #os.system("convert -crop 590x272+0+90 +repage " + sys.argv[1] + imglst[i] + " " + sys.argv[2] + imglst[i])

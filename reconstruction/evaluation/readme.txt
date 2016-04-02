###########################################################################
#   THE KITTI VISION BENCHMARK SUITE: STEREO AND OPTICAL FLOW BENCHMARK   #
#              Andreas Geiger    Philip Lenz    Raquel Urtasun            #
#                    Karlsruhe Institute of Technology                    #
#                Toyota Technological Institute at Chicago                #
#                             www.cvlibs.net                              #
###########################################################################

This file describes the KITTI stereo and optical flow benchmarks,
consisting of 194 training and 195 test image pairs for each task. Ground
truth has been aquired by accumulating 3D point clouds from a 360 degree
Velodyne HDL-64 Laserscanner. Note that for the optical flow benchmark,
making use of epipolar constraints and/or using stereo information is
forbidden!

NOTE: WHEN SUBMITTING RESULTS, PLEASE STORE THEM IN THE SAME DATA FORMAT IN
WHICH THE GROUND TRUTH DATA IS PROVIDED (SEE BELOW), USING THE FILE NAMES
000000_10.png TO 000194_10.png. CREATE A ZIP ARCHIVE OF THEM AND STORE
YOUR RESULTS IN ITS ROOT FOLDER.

File description:
=================

The folders testing and training contain the grayscale video images in
the sub-folders image_0 (left image) and image_1 (right image). All input
images are saved as unsigned char greyscale PNG images. Filenames are
composed of a 6-digit image index as well as a 2-digit frame number:

 - xxxxxx_yy.png

Here xxxxxx is running from 0 to 193/194 for the training/test dataset and
the frame number yy is either 10 or 11. The reference images, for which
results must be provided, are the left image of frame 10 for each test pair.

Corresponding ground truth disparity maps and flow fields can be found in
the folders disp and flow, respectively. Here the suffix _noc or _occ refers
to non-occluded or occluded (=all pixels).

File naming examples:
=====================

Test stereo pair '000005':

 - left image:  testing/image_0/000005_10.png
 - right image: testing/image_1/000005_10.png

Test flow pair '000005':

 - first frame:  testing/image_0/000005_10.png
 - second frame: testing/image_0/000005_11.png

Data format:
============

Disparity and flow values range [0..256] and [-512..+512] respectively. For
both image types documented MATLAB and C++ utility functions are provided
within this development kit in the folders matlab and cpp. If you want to
use your own code instead, you can use the following guidelines:

Disparity maps are saved as uint16 PNG images, which can be opened with
either MATLAB or libpng++. A 0-value indicates that no ground truth exists
for that pixel. Otherwise the disparity for a pixel can be computed by
converting the uint16 value to float and dividing it by 256:

disp(u,v)  = ((float)I(u,v))/256.0;
valid(u,v) = I(u,v)>0;

Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
contains the u-component, the second channel the v-component and the third
channel denotes if a valid ground truth optical flow value exists for that
pixel (1 if true, 0 otherwise). To convert the u-/v-flow into floating point
values, convert the value to float, subtract 2^15 and divide the result by 64:

flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0;
flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0;
valid(u,v)  = (bool)I(u,v,3);

Evaluation Code:
================

For transparency we have included the KITTI evaluation code in the
subfolder 'cpp' of this development kit. It can be compiled via:

g++ -O3 -DNDEBUG -o evaluate_stereo evaluate_stereo.cpp -lpng
g++ -O3 -DNDEBUG -o evaluate_flow evaluate_flow.cpp -lpng


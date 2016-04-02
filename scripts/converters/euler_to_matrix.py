#!/usr/bin/python
import numpy as np
import transformations

import np_helper

#use Projects/modules/...py for rotation matrix


pts = np.array(np.loadtxt('/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/groundtruth/camera_track_numpy.txt'))

fp = open("tsukuba_gt.txt", "w")

for i in range(len(pts)):
    # we need to invert the y and z axis because Tsukuba has a strange coordinate system
    pts[i][3] = np.radians(pts[i][3])
    pts[i][4] = np.radians(-pts[i][4])
    pts[i][5] = np.radians(-pts[i][5])
    Rt = transformations.euler_matrix(pts[i][3], pts[i][4], pts[i][5])
    #Rt = np.hstack((R, np.array([[pts[i][3]], [pts[i][4]], [pts[i][5]]])))
    Rt[0,3] = pts[i][0]
    Rt[1,3] = -pts[i][1]
    Rt[2,3] = -pts[i][2]
    #print("Rt before = ", Rt)
    #Rt = np_helper.inv_Rt(Rt) - NO
    #print("Rt after = ", Rt)
    for j in range(3):
        for k in range(4):
            fp.write("%f " % Rt[j,k])
    fp.write("\n")

    #fp.write("1 0 0 %f 0 1 0 %f 0 0 1 %f\n" % (pts[i][0], -pts[i][1], -pts[i][2]))

fp.close()

#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math

filepath1 = "/home/kivan/Projects/datasets/KITTI/poses/07.txt"
filepath2 = "/home/kivan/Dropbox/experiment_data/img/07_nodf.txt"
#filepath2 = "/home/kivan/Dropbox/experiment_data/img/07_df.txt"
#filepath2 = "/home/kivan/Dropbox/experiment_data/img/07_wgt.txt"
#filepath1 = "/home/kivan/Projects/cv-stereo/data/GT/Tsukuba/tsukuba_gt_crop.txt"
#filepath2 = "/home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/results/tsukuba_tracker_ncc_best/00.txt"
#filepath2 = "/home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/results/tsukuba_tracker_ncc_best_df/00.txt"

gt_pts = np.array(np.loadtxt(filepath1))
vo_pts = np.array(np.loadtxt(filepath2))

if gt_pts.shape[0] != vo_pts.shape[0]:
    print("GT and VO data not the same size\n")
    exit(-1)
gt_pts3D = np.zeros((gt_pts.shape[0], 3))
vo_pts3D = np.zeros((vo_pts.shape[0], 3))
for i in range(len(vo_pts)):
   vo_pts3D[i,0] = vo_pts[i,3]
   vo_pts3D[i,1] = vo_pts[i,7]
   vo_pts3D[i,2] = vo_pts[i,11]
   gt_pts3D[i,0] = gt_pts[i,3]
   gt_pts3D[i,1] = gt_pts[i,7]
   gt_pts3D[i,2] = gt_pts[i,11]

fig_path = plt.figure(figsize=(14,8))
plt.axes().set_aspect('equal')

plt.plot(gt_pts3D[:,0], gt_pts3D[:,2], color='r', label="GT")
plt.plot(gt_pts3D[::100,0], gt_pts3D[::100,2], marker='.', color='k', ls="")
plt.plot(gt_pts3D[0,0], gt_pts3D[0,2], marker='o', color='r', ls="")
plt.plot(vo_pts3D[:,0], vo_pts3D[:,2], color='b', label="VO")
plt.plot(vo_pts3D[::100,0], vo_pts3D[::100,2], marker='.', color='k', ls='')
plt.plot(vo_pts3D[0,0], vo_pts3D[0,2], marker='o', color='b', ls="")
#for i in range(0,len(vo_pts3D),10):
#  #plt.text(vo_pts3D[i,0]+2, vo_pts3D[i,2]+2, str(i), color='b')
#  plt.text(gt_pts3D[i,0]+2, gt_pts3D[i,2]+2, str(i), color='r')
plt.xlabel("x (m)", fontsize=26)
plt.ylabel("z (m)", fontsize=26)
plt.legend(loc="upper right", fontsize=30)
#plt.title(filepath1+"\n"+filepath2, fontsize=12)
plt.xlim([-200, 20])
plt.ylim([-100, 130])

fig_path.savefig("plot_path_diff.pdf", bbox_inches='tight')
plt.show()


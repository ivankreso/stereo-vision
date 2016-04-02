#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt


#gt_pts = np.array(np.loadtxt('/home/kivan/Projects/datasets/KITTI/poses/07.txt'))
gt_pts = np.array(np.loadtxt('vo_kitti_07_bfm_refined.txt'))

#vo_pts = np.array(np.loadtxt('vo_kitti_07_libviso_orig.txt'))
vo_pts = np.array(np.loadtxt('vo_kitti_07_libviso_refined.txt'))
#vo_pts = np.array(np.loadtxt('vo_kitti_07_bfm_refined.txt'))

fig_path1 = plt.figure(figsize=(14,8))
plt.axes().set_aspect('equal')

plt.plot(gt_pts[:,3], gt_pts[:,11], color='r')
plt.plot(gt_pts[::100,3], gt_pts[::100,11], marker='.', color='k', ls="")
plt.plot(gt_pts[0,3], gt_pts[0,11], marker='o', color='r', ls="")

#plt.plot(kitti_pts[:,0], kitti_pts[:,1], color='r')
#plt.plot(kitti_pts[::100,0], kitti_pts[::100,1], marker='.', color='k', ls="")
#plt.plot(kitti_pts[0,0], kitti_pts[0,1], marker='o', color='r', ls="")

plt.plot(vo_pts[:,3], vo_pts[:,11], color='b')
plt.plot(vo_pts[::100,3], vo_pts[::100,11], marker='.', color='k', ls='')
plt.plot(vo_pts[0,3], vo_pts[0,11], marker='o', color='b', ls="")

#for i in range(0, vo_pts.shape[0], 5):
#   #plt.text(vo_pts[i,3], vo_pts[i,11], str(vo_pts[i,7]), color='b')
#   plt.text(vo_pts[i,3], vo_pts[i,11], '{0:.{1}f}'.format(vo_pts[i,7], 1) + " (" + str(i) + ")", color='b')
##  plt.text(gps_pts[i,0]+2, gps_pts[i,1]+2, str(i), color='r')


plt.show()
fig_path1.savefig('plot_path.pdf', bbox_inches='tight')

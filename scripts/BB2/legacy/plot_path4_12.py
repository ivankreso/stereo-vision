#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

gt_pts = np.array(np.loadtxt('/home/kreso/projects/master_thesis/datasets/KITTI/odometry_pose/poses/00.txt'));
vo_pts = np.array(np.loadtxt('/home/kreso/projects/master_thesis/src/stereo-master/scripts/path_plotter/viso_points_dataset00.txt'));


fig_path = plt.figure()
plt.axes().set_aspect('equal')
plt.plot(gt_pts[:,3], gt_pts[:,11], color='r')
plt.plot(gt_pts[::100,3], gt_pts[::100,11], marker='.', color='k', ls="")
plt.plot(gt_pts[0,3], gt_pts[0,11], marker='o', color='r', ls="")
plt.plot(vo_pts[:,0], vo_pts[:,2], color='b')
plt.plot(vo_pts[::100,0], vo_pts[::100,2], marker='.', color='k', ls='')
plt.plot(vo_pts[0,0], vo_pts[0,2], marker='o', color='r', ls="")
plt.show()

fig_path.savefig("plot_kitti_00.png", bbox_inches='tight', dpi=300)


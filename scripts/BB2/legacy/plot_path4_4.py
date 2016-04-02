#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

gt_pts = np.array(np.loadtxt('viso_points_550.txt'));
vo_pts = np.array(np.loadtxt('viso_points_bb.txt'));


plt.figure()
plt.axes().set_aspect('equal')
plt.plot(gt_pts[:,0], gt_pts[:,2], color='r')
plt.plot(gt_pts[::100,0], gt_pts[::100,2], marker='.', color='k', ls="")
plt.plot(gt_pts[0,0], gt_pts[0,2], marker='o', color='r', ls="")
plt.plot(vo_pts[:,0], vo_pts[:,2], color='b')
plt.plot(vo_pts[::100,0], vo_pts[::100,2], marker='.', color='k', ls='')
plt.plot(vo_pts[0,0], vo_pts[0,2], marker='o', color='r', ls="")
plt.show()

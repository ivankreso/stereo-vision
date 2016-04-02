#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

font_size = 28 # 32 for 23" lcd

path1 = np.array(np.loadtxt('viso_points_sim_170m_base_0.09.txt'))
path2 = np.array(np.loadtxt('viso_points_sim_170m_base_0.12.txt'))
path3 = np.array(np.loadtxt('viso_points_sim_170m_base_0.20.txt'))
path4 = np.array(np.loadtxt('viso_points_sim_170m_base_0.50.txt'))
#path5 = np.array(np.loadtxt('viso_points_sim_170m_base_0.50.txt'))
path_gt = np.array(np.loadtxt('viso_points_sim_170m_gt.txt'))


#fig_path = plt.figure(figsize=(10,5), dpi=100)
fig_path = plt.figure(figsize=(10,5))

#plt.axes().set_aspect('equal')
plt.axis([0, 210, 0, 450], "equal")

scale_x = 600

path_gt[:,3] += 370
plt.plot(path_gt[:,11], path_gt[:,3], color='k', linewidth=3, label="groundtruth")
#plt.plot(path_gt[::100,11], path_gt[::100,3], marker='.', color='k', ls="")
#plt.plot(path_gt[0,11], path_gt[0,3], marker='o', color='k', ls="")


#path7[:,3] += 0.5 * scale_x
#plt.plot(path7[:,11], path7[:,3], color='#655252', linewidth=3, label="baseline = 0.5")


#path6[:,3] += 0.3 * scale_x
#plt.plot(path6[:,11], path6[:,3], color='m', linewidth=3, label="baseline = 0.3")


#path5[:,3] += 0.2 * scale_x
#plt.plot(path5[:,11], path5[:,3], color='c', linewidth=3, label="baseline = 0.2")


path4[:,3] += 290
plt.plot(path4[:,11], path4[:,3], color='m', linewidth=3, label="b = 0.5")

path3[:,3] += 210
plt.plot(path3[:,11], path3[:,3], color='r', linewidth=3, label="b = 0.2")

path2[:,3] += 130
plt.plot(path2[:,11], path2[:,3], color='g', linewidth=3, label="b = 0.12")

path1[:,3] += 50
plt.plot(path1[:,11], path1[:,3], color='b', linewidth=3, label="b = 0.09")

plt.axes().set_yticks(ticks=[])

#plt.plot(gps_pts[:,0], gps_pts[:,1], color='g')
#plt.plot(gps_pts[::100,0], gps_pts[::100,1], marker='.', color='k', ls='')
#plt.plot(gps_pts[0,0], gps_pts[0,1], marker='o', color='g', ls="")

plt.xlabel('Z (m)', fontsize=font_size)
plt.ylabel('X (m)', fontsize=font_size)
plt.legend(loc='lower right', fontsize=font_size-4)
#plt.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()
fig_path.savefig("plot_baseline_scale.pdf", bbox_inches='tight')


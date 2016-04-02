#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math

filepath = "/home/kivan/source/cv-stereo/results/block3_igor.txt"

egomotions = np.array(np.loadtxt(filepath))

pts3d = np.zeros((egomotions.shape[0], 3))
for i in range(egomotions.shape[0]):
    pts3d[i,0] = egomotions[i,3]
    pts3d[i,1] = egomotions[i,7]
    pts3d[i,2] = egomotions[i,11]

fig_path = plt.figure(figsize=(14,8))
plt.axes().set_aspect('equal')

plt.plot(pts3d[:,0], pts3d[:,2], color='b')

#plt.plot(gt_pts3D[::100,0], gt_pts3D[::100,2], marker='.', color='k', ls="")
#plt.plot(gt_pts3D[0,0], gt_pts3D[0,2], marker='o', color='r', ls="")

plt.xlabel("x (m)", fontsize=26)
plt.ylabel("z (m)", fontsize=26)
#plt.title(filepath, fontsize=12)
#plt.legend(loc="upper left", fontsize=22)

distance = 0.0
for i in range(egomotions.shape[0] - 1):
    distance += np.linalg.norm(pts3d[i,:] - pts3d[i+1,:])
print("Traveled distance = ", distance)

fig_path.savefig("path.pdf", bbox_inches='tight')
#plt.show()

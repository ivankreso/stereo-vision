import numpy as np
import matplotlib.pyplot as plt

error_gt_refiner = np.array(np.loadtxt('reproj_error_gt_refiner.txt'))
error_gt = np.array(np.loadtxt('reproj_error_gt.txt'))
error_vo_refiner = np.array(np.loadtxt('reproj_error_refiner.txt'))
error_vo = np.array(np.loadtxt('reproj_error.txt'))

# plot errors
fig_scale = plt.figure(figsize=(12,8))
plt.plot(error_gt, marker='o', color='r', label="without KLT refiner")
plt.plot(error_gt_refiner, marker='o', color='b', label="with KLT refiner")
#plt.plot([0,120], [1.0,1.0], ls="--", color="k")
#fig_scale.suptitle('Scale error', fontsize=18)
plt.xlabel('time frames', fontsize=30)
plt.ylabel('mean reprojection error', fontsize=30)
plt.legend(fontsize=24)

fig_scale = plt.figure(figsize=(12,8))
plt.plot(error_gt, marker='o', color='r', label="GT")
plt.plot(error_gt_refiner, marker='o', color='g', label="GT refined")
plt.plot(error_vo, marker='o', color='b', label="VO")
plt.plot(error_vo_refiner, marker='o', color='k', label="VO refined")
#plt.plot([0,120], [1.0,1.0], ls="--", color="k")
#fig_scale.suptitle('Scale error', fontsize=18)
plt.xlabel('time frames', fontsize=30)
plt.ylabel('mean reprojection error', fontsize=30)
plt.legend(fontsize=24)

plt.show()


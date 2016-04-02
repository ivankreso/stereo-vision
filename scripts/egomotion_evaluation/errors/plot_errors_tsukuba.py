import numpy as np
import matplotlib.pyplot as plt

#error1 = np.array(np.loadtxt('depth_error_tsukuba_bfm.txt'))
#error1 = np.array(np.loadtxt('depth_error_tsukuba_libviso.txt'))
#error1 = np.array(np.loadtxt('depth_error_tsukuba_libviso_refiner.txt'))
error1 = np.array(np.loadtxt('depth_error_tsukuba_bfm.txt'))
#error1 = np.array(np.loadtxt('depth_mae.txt'))
#error2 = np.array(np.loadtxt('depth_error_tsukuba_libviso_subpixel.txt'))
error2 = np.array(np.loadtxt('depth_error_bfm_smoothing.txt'))
#error2 = np.array(np.loadtxt('depth_mae_subpixel.txt'))
#error2 = np.array(np.loadtxt('depth_error_tsukuba_bfm_refiner.txt'))
#error_vo_refiner = np.array(np.loadtxt('reproj_error_refiner.txt'))
#error_vo = np.array(np.loadtxt('reproj_error.txt'))

diff = error1 - error2
print(np.sum(diff) / np.size(diff))

# plot errors
fig_scale = plt.figure(figsize=(12,8))
plt.plot(error1, marker='o', color='r', label="without KLT refiner")
plt.plot(error2, marker='o', color='b', label="with KLT refiner")
#plt.plot([0,120], [1.0,1.0], ls="--", color="k")
#fig_scale.suptitle('Scale error', fontsize=18)
plt.xlabel('time frames', fontsize=30)
plt.ylabel('mean depth error', fontsize=30)
plt.legend(fontsize=24)

#fig_scale = plt.figure(figsize=(12,8))
#plt.plot(error_gt, marker='o', color='r', label="GT")
#plt.plot(error_gt_refiner, marker='o', color='g', label="GT refined")
#plt.plot(error_vo, marker='o', color='b', label="VO")
#plt.plot(error_vo_refiner, marker='o', color='k', label="VO refined")
##plt.plot([0,120], [1.0,1.0], ls="--", color="k")
##fig_scale.suptitle('Scale error', fontsize=18)
#plt.xlabel('time frames', fontsize=30)
#plt.ylabel('mean reprojection error', fontsize=30)
#plt.legend(fontsize=24)

plt.show()


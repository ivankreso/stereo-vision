#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math

filepath1 = "/home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/results/bb2_tracker_freak_7_1/bb.txt"
filepath2 = "/home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/results/bb2_tracker_freak_7_1_ba/bb.txt"
#filepath1 = "/home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/results/bb2_tracker_freak_7_2/bb.txt"
#filepath2 = "/home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/results/bb2_tracker_freak_7_2_ba/bb.txt"

gt_pts = np.array(np.loadtxt(filepath1))

#gt_pts = np.array(np.loadtxt('/home/kivan/Projects/datasets/KITTI/poses/07.txt'))
#gt_pts = np.array(np.loadtxt('/home/kivan/Projects/datasets/KITTI/poses/00.txt'))
#gt_pts = np.array(np.loadtxt('gt_track_tsukuba_crop.txt'))
#gt_pts = np.array(np.loadtxt('vo_bb_libvisotracker.txt'))
#gt_pts = np.array(np.loadtxt('vo_bb_bfm.txt'))
#gt_pts = np.array(np.loadtxt('vo_07_libvisotracker.txt'))
#gt_pts = np.array(np.loadtxt('vo_07_bfm.txt'))
#gt_pts = np.array(np.loadtxt('vo_tsukuba_bfm.txt'))
#gt_pts = np.array(np.loadtxt('vo_tsukuba_libviso_refiner.txt'))

vo_pts = np.array(np.loadtxt(filepath2))

#vo_pts = np.array(np.loadtxt('/home/kivan/Projects/cv-stereo/build/vo_sba/release/vo.txt'))
#vo_pts = np.array(np.loadtxt('/home/kivan/Projects/cv-stereo/build/vo_sba/release/sba_vo.txt'))
#vo_pts = np.array(np.loadtxt('vo_00_bfm.txt'))
#vo_pts = np.array(np.loadtxt('vo_00_libviso.txt'))
#vo_pts = np.array(np.loadtxt('vo_tsukuba_libviso.txt'))
#vo_pts = np.array(np.loadtxt('vo_tsukuba_libviso_subpixel.txt'))
#vo_pts = np.array(np.loadtxt('vo_tsukuba_bfm_refiner.txt'))
#vo_pts = np.array(np.loadtxt('vo_tsukuba_libviso.txt'))
#vo_pts = np.array(np.loadtxt('vo_tsukuba_libviso_refiner.txt'))
#vo_pts = np.array(np.loadtxt('vo_tsukuba_libvisotracker_refinernew.txt'))
#vo_pts = np.array(np.loadtxt('vo_bb_libvisotracker_refiner.txt'))
#vo_pts = np.array(np.loadtxt('vo_bb_bfm_refiner.txt'))
#vo_pts = np.array(np.loadtxt('vo_07_libvisotracker_refiner.txt'))
#vo_pts = np.array(np.loadtxt('vo_07_bfm_refiner.txt'))

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

plt.plot(gt_pts3D[:,0], gt_pts3D[:,2], color='r')
plt.plot(gt_pts3D[::100,0], gt_pts3D[::100,2], marker='.', color='k', ls="")
plt.plot(gt_pts3D[0,0], gt_pts3D[0,2], marker='o', color='r', ls="")
plt.plot(vo_pts3D[:,0], vo_pts3D[:,2], color='b')
plt.plot(vo_pts3D[::100,0], vo_pts3D[::100,2], marker='.', color='k', ls='')
plt.plot(vo_pts3D[0,0], vo_pts3D[0,2], marker='o', color='b', ls="")
#for i in range(0,len(vo_pts3D),10):
#  #plt.text(vo_pts3D[i,0]+2, vo_pts3D[i,2]+2, str(i), color='b')
#  plt.text(gt_pts3D[i,0]+2, gt_pts3D[i,2]+2, str(i), color='r')
plt.xlabel("x (m)", fontsize=26)
plt.ylabel("z (m)", fontsize=26)
plt.legend(loc="upper left", fontsize=22)
plt.title(filepath1+"\n"+filepath2, fontsize=12)

plt.show()
exit()

#for i in range(0, vo_pts3D.shape[0], 5):
#   #plt.text(vo_pts3D[i,3], vo_pts3D[i,11], str(vo_pts3D[i,7]), color='b')
#   plt.text(vo_pts3D[i,3], vo_pts3D[i,11], '{0:.{1}f}'.format(vo_pts3D[i,7], 1) + " (" + str(i) + ")", color='b')
##  plt.text(gt_pts3D[i,0]+2, gt_pts3D[i,1]+2, str(i), color='r')

# angle between 2 vectors defined by 3 points using dot product
def calcphi(pt1,pt2,pt3):
   v1=pt2-pt1
   v2=pt3-pt2
   return math.degrees(math.acos(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)))

# angle between 2 vectors using vector product (-90, 90)
def calcphi2vec(v1,v2):
   return math.degrees(math.asin(np.linalg.norm(np.cross(v1, v2)) / np.linalg.norm(v1)/np.linalg.norm(v2)))

def calcphi2(pt1,pt2,pt3):
   v1=pt2-pt1
   v2=pt3-pt2
   return calcphi2vec(v1,v2)

# angular movement data
gt_phis=np.array([calcphi2(gt_pts3D[i-1],gt_pts3D[i],gt_pts3D[i+1]) for i in range(1,len(gt_pts3D)-1)])
vo_phis=np.array([calcphi2(vo_pts3D[i-1],vo_pts3D[i],vo_pts3D[i+1]) for i in range(1,len(vo_pts3D)-1)])
# angular movement difference between gps od visual odometry
# cant do this before vo and gps paths are not mapped with starting point and rotation offset
#gps_vo_phis=[calcphi2vec(gt_pts3D[i]-gt_pts3D[i-1], vo_pts3D[i]-vo_pts3D[i-1]) for i in range(1,len(vo_pts3D))]

# speed movement data
gps_speed=np.array([np.linalg.norm(gt_pts3D[i]-gt_pts3D[i-1]) for i in range(1,len(vo_pts3D))])
vo_speed=np.array([np.linalg.norm(vo_pts3D[i]-vo_pts3D[i-1]) for i in range(1,len(vo_pts3D))])

#print (gt_phis[0:10])
#print (vo_phis[0:10])
#print([gt_pts3D[i] for i in range(0,10)])
#print([vo_pts3D[i] for i in range(0,10)])
#print([vo_pts3D[i]-vo_pts3D[i-1] for i in range(1,10)])
#print(calcphi(vo_pts3D[2-2],vo_pts3D[2-1],vo_pts3D[2]))
#plt.plot(gt_pts3D[:10,0], gt_pts3D[:10,1], marker='o', color='r')
#plt.plot(vo_pts3D[:10,0], vo_pts3D[:10,1], marker='o', color='b')

trans_mse = np.mean(np.square(gps_speed - vo_speed))
trans_mae = np.mean(np.abs(gps_speed - vo_speed))
print("translation error MSE: ", trans_mse)
print("translation error MAE: ", trans_mae)
fig_speed = plt.figure(figsize=(12,8))
plt.plot(range(1,len(vo_pts3D)), gps_speed, marker='o', color='r', label="GPS")
plt.plot(range(1,len(vo_pts3D)), vo_speed, marker='o', color='b', label="visual odometry")
plt.title("MSE = " + str(trans_mse)[:5] + ",  MAE = " + str(trans_mae)[:5], fontsize=30)
#plt.title('Speed', fontsize=14)
plt.xlabel('time (s)', fontsize=30)
plt.ylabel('distance (m)', fontsize=30)
plt.legend(fontsize=24)

# plot scale error of visual odometry
fig_scale = plt.figure(figsize=(12,8))
scale_err = np.array(gps_speed) / np.array(vo_speed)
plt.plot(scale_err, marker='o', color='r')
plt.plot([0,120], [1.0,1.0], ls="--", color="k")
#fig_scale.suptitle('Scale error', fontsize=18)
plt.xlabel('time (s)', fontsize=30)
plt.ylabel('scale error (gps / odometry)', fontsize=30)

#print(gt_phis)
#print(vo_phis)
#print(np.square(gt_phis - vo_phis))
#print((gt_phis - vo_phis))
#print(np.square(gt_phis - vo_phis))
rot_mse = np.mean(np.square(gt_phis - vo_phis))
rot_mae = np.mean(np.abs(gt_phis - vo_phis))
print("rotation error MSE: ", rot_mse)
print("rotation error MAE: ", rot_mae)
fig_rot = plt.figure(figsize=(12,8))
plt.plot(range(1,len(vo_pts3D)-1), gt_phis, marker='o', color='r', label="GPS rotation angles")
plt.plot(range(1,len(vo_pts3D)-1), vo_phis, marker='o', color='b', label="odometry rotation angles")
#plt.plot(range(1,len(vo_pts3D)-1), gps_vo_phis[:-1], marker='o', color='b', label="TODO")
plt.xlabel('time (s)', fontsize=26)
plt.ylabel('angle (deg)', fontsize=26)
#plt.text(45, 20, "average error = " + str(rot_avgerr)[:5], color='b', fontsize=16)
plt.title("MSE = " + str(rot_mse)[:5] + ", MAE = " + str(rot_mae)[:5], fontsize=26)
plt.legend(fontsize=22)

fig_path.savefig("plot_path_diff.pdf", bbox_inches='tight')
fig_speed.savefig("plot_speed.pdf", bbox_inches='tight')
fig_scale.savefig('plot_scale_error.pdf', bbox_inches='tight')
fig_rot.savefig("plot_rotation_diff.pdf", bbox_inches='tight')

plt.show()

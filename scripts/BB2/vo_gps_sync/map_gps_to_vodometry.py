#!/usr/bin/python
import os
import math
import re
import struct
import numpy as np
import matplotlib.pyplot as plt

def readstamp(f):
   pgmoffset=17
   bs=f.read(pgmoffset+4)
   x=struct.unpack("<I", bs[pgmoffset:pgmoffset+4])[0]    # reverse byte reading order
   t = (x>>0)  & 0xffffffff
   t = ((t >> 16) & 0xffff) | ((t << 16) & 0xffff0000)
   secs = (t >> 25) & 0x7f
   cycles = (t >> 12) & 0x1fff
   offset = (t >> 0) & 0xfff
   return secs + ((cycles + (offset / 3072.0)) / 8000.0)

def getTime(gps_pt):
   return gps_pt[10]*60 + gps_pt[11]


gps_pts = np.array(np.loadtxt('gps_data.txt'))
#vo_pts = np.array(np.loadtxt('viso_points_calib2_fixed_libviso.txt'))
#vo_pts = np.array(np.loadtxt('viso_points_calib1_orig_libviso.txt'))
#vo_pts = np.array(np.loadtxt('vo_bb_mytracker.txt'))
#vo_pts = np.array(np.loadtxt('vo_bb_libviso_subpixel.txt'))
#vo_pts = np.array(np.loadtxt("/home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/results/bb2_tracker_freak_7_1/bb.txt"))
vo_pts = np.array(np.loadtxt("/home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/results/bb2_tracker_freak_7_2/bb.txt"))
#vo_pts = np.array(np.loadtxt('vo_bb_libvisotracker_refiner.txt'))
#vo_pts = np.array(np.loadtxt('vo_bb_bfm.txt'))
#vo_pts = np.array(np.loadtxt('vo_bb_bfm_refiner.txt'))

#vo_pts = np.array(np.loadtxt('viso_points_calib2_orig_libviso.txt'))
#vo_pts = np.array(np.loadtxt('viso_points_orig.txt'));
src_folder = '/home/kivan/Projects/datasets/bumblebee/20121031/'
#src_folder = '/home/kreso/projects/master_thesis/datasets/bumblebee_new/'
vo_times=[]
t_prev = 0
cycles = 0
for name in sorted(os.listdir(src_folder)):
   m=re.match(r'fc2.*pgm', name)
   if m:
      t = readstamp(open(src_folder+name, mode='rb'))
      if t < t_prev:
         cycles += 1
      time = t + (cycles * 128.0)
      t_prev = t
      #print('{} {}'.format(name, time))
      vo_times.append(time)

t0 = vo_times[0]
vo_times = [t - t0 for t in vo_times]
#print(vo_times)

# set odometry start time (0 is start time of first gps point)
vo_start = 3.05 # 2.8 3.3 3.0
vo_times = [t + vo_start for t in vo_times]
#print(vo_times)

np.savetxt("times.txt", vo_times)

# we use every 3 frames in odometry
print("Number of frames: ", len(vo_times))
vo_times=vo_times[::3]
vo_pts=vo_pts[::]
print("Number of frames after sampling: ", len(vo_times))

vo_pts2D=np.ndarray((vo_pts.shape[0], 2))
vo_pts3D=np.zeros((vo_pts.shape[0], 3))
for i in range(len(vo_pts)):
   vo_pts3D[i,0]=vo_pts[i,3]
   vo_pts3D[i,1]=vo_pts[i,11]
   vo_pts3D[i,2]=vo_pts[i,7]     # Y_vo <-> Z_gps (alt)

# first point time of gps must be bigger then vis. odo. start time
# otherwise we dont have data to interpolate it
# print(len(gps_pts), gps_pts.shape, len(vo_pts))
t0 = getTime(gps_pts[0])
for i in range(len(gps_pts)):
   # cut and break in first gps point with bigger time
   print("skip")
   if getTime(gps_pts[i])-t0 > vo_times[0]:
      gps_pts=gps_pts[i:]
      break

# interpoliramo vizualnu odometriju u vremenima
# točaka GPS-a
vo_inter = np.zeros((gps_pts.shape[0], 3))
for i in range(len(gps_pts)):
   pt = gps_pts[i]
   t = getTime(pt) - t0
   #print(t)
   for j in range(len(vo_pts3D)):
      if vo_times[j] >= t:
         if i == 0:
            vo_pts_crop = vo_pts3D[j-1:,:]
         assert j>0
         # print(" -> ", vo_times[j])
         alfa = (t - vo_times[j-1]) / (vo_times[j] - vo_times[j-1])
         vo_inter[i] = (1-alfa) * vo_pts3D[j-1] + alfa * vo_pts3D[j]
         # print(i, vo_inter[i])
         break
   else:
      vo_inter=vo_inter[:i,:]
      gps_pts=gps_pts[:i,:]
      break

gps_pts = gps_pts[:,0:3]
#print(gps_pts)
#print(vo_pts2D)
#print(vo_inter)

#plt.plot(vo_pts2D[:,0], vo_pts2D[:,1], marker='.', color='r', label="VO_orig")
#plt.plot(vo_pts_crop[:,0], vo_pts_crop[:,1], marker='.', color='b', label="VO_orig")
#plt.plot(vo_inter[:,0], vo_inter[:,1], marker='.', color='b', label="VO_inter")
#plt.show()
#exit(0)

# angle between 2 vectors defined by 3 points using dot product
def calcphi(pt1,pt2,pt3):
   v1=pt2-pt1
   v2=pt3-pt2
   return math.degrees(math.acos(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)))

# angle between 2 vectors using vector product (-90, 90)
def calcphi2vec(v1,v2):
   return math.degrees(math.asin(np.linalg.norm(np.cross(v1, v2))/
            np.linalg.norm(v1)/np.linalg.norm(v2)))

def calcphi2(pt1,pt2,pt3):
   v1=pt2-pt1
   v2=pt3-pt2
   return calcphi2vec(v1,v2)

# angular movement data
gps_phis=np.array([calcphi2(gps_pts[i-1],gps_pts[i],gps_pts[i+1]) for i in range(1,len(vo_inter)-1)])
vo_phis=np.array([calcphi2(vo_inter[i-1],vo_inter[i],vo_inter[i+1]) for i in range(1,len(vo_inter)-1)])
# angular movement difference between gps od visual odometry
# cant do this before vo and gps paths are not mapped with starting point and rotation offset
#gps_vo_phis=[calcphi2vec(gps_pts[i]-gps_pts[i-1], vo_inter[i]-vo_inter[i-1]) for i in range(1,len(vo_inter))]

# speed movement data
gps_speed=np.array([np.linalg.norm(gps_pts[i]-gps_pts[i-1]) for i in range(1,len(vo_inter))])
vo_speed=np.array([np.linalg.norm(vo_inter[i]-vo_inter[i-1]) for i in range(1,len(vo_inter))])

#print (gps_phis[0:10])
#print (vo_phis[0:10])
#print([gps_pts[i] for i in range(0,10)])
#print([vo_inter[i] for i in range(0,10)])
#print([vo_inter[i]-vo_inter[i-1] for i in range(1,10)])
#print(calcphi(vo_inter[2-2],vo_inter[2-1],vo_inter[2]))
#plt.plot(gps_pts[:10,0], gps_pts[:10,1], marker='o', color='r')
#plt.plot(vo_inter[:10,0], vo_inter[:10,1], marker='o', color='b')

trans_mse = np.mean(np.square(gps_speed - vo_speed))
trans_mae = np.mean(np.abs(gps_speed - vo_speed))
print("translation error MSE: ", trans_mse)
print("translation error MAE: ", trans_mae)
fig_speed = plt.figure(figsize=(12,8))
plt.plot(range(1,len(vo_inter)), gps_speed, marker='o', color='r', label="GPS")
plt.plot(range(1,len(vo_inter)), vo_speed, marker='o', color='b', label="visual odometry")
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

#print(gps_phis)
#print(vo_phis)
#print(np.square(gps_phis - vo_phis))
#print((gps_phis - vo_phis))
#print(np.square(gps_phis - vo_phis))
rot_mse = np.mean(np.square(gps_phis - vo_phis))
rot_mae = np.mean(np.abs(gps_phis - vo_phis))
print("rotation error MSE: ", rot_mse)
print("rotation error MAE: ", rot_mae)

fig_rot = plt.figure(figsize=(12,8))
plt.plot(range(1,len(vo_inter)-1), gps_phis, marker='o', color='r', label="GPS rotation angles")
plt.plot(range(1,len(vo_inter)-1), vo_phis, marker='o', color='b', label="odometry rotation angles")
#plt.plot(range(1,len(vo_inter)-1), gps_vo_phis[:-1], marker='o', color='b', label="TODO")
plt.xlabel('time (s)', fontsize=26)
plt.ylabel('angle (deg)', fontsize=26)
#plt.text(45, 20, "average error = " + str(rot_avgerr)[:5], color='b', fontsize=16)
plt.title("MSE = " + str(rot_mse)[:5] + ", MAE = " + str(rot_mae)[:5], fontsize=26)
plt.legend(fontsize=22)

fig_path = plt.figure(figsize=(8,8))
#plt.axis('equal')
#plt.axis([-200, 100, -50, 250], 'equal')
plt.axis([-50, 200, -100, 150], 'equal')

#gps_pts[:,1] += 40.0
# translate gps to (0,0)
gps_pts[:,0] -= gps_pts[0,0]
gps_pts[:,1] -= gps_pts[0,1]
gps_pts[:,2] -= gps_pts[0,2]
vo_inter[:,0] -= vo_inter[0,0]
vo_inter[:,1] -= vo_inter[0,1] + 1.3
vo_inter[:,2] -= vo_inter[0,2]
vo_pts_crop[:,0] -= vo_pts_crop[0,0]
vo_pts_crop[:,1] -= vo_pts_crop[0,1] + 1.3
vo_pts_crop[:,2] -= vo_pts_crop[0,2]

#np.savetxt("gps_pts.txt", gps_pts)
#np.savetxt("vo_inter_pts.txt", vo_inter)

# legacy rotation just for pretty display
angle = -2.02 #-2.02
#angle = -2.1  # alan calib
R_gps = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
# rotate just for graph
#vo_inter[:,0:2] = R_gps.dot(vo_inter[:,0:2].T);
#vo_inter = vo_inter.T

gps_pts = R_gps.dot(gps_pts[:,0:2].T);
gps_pts = gps_pts.T

# optimal rotation (N=4) for system calib
#R_vo = np.array([[-0.3870,   -0.9200,    0.0610],
#                 [0.9220,   -0.3869,    0.0152],
#                 [0.0096,    0.0621,    0.9980]]);
#
#vo_inter = R_vo.dot(vo_inter.T);
#vo_inter = vo_inter.T

#np.savetxt("gps_pts_calib.txt", gps_pts)
#np.savetxt("vo_inter_pts_calib.txt", vo_inter)


print(gps_pts.shape)
print(vo_pts3D.shape)

gps_pts = np.vstack((gps_pts, gps_pts[0,:]))
plt.plot(gps_pts[:,0], gps_pts[:,1], marker='.', color='r', label="GPS")
plt.plot(gps_pts[::5,0], gps_pts[::5,1], marker='.', color='k', ls="")
plt.plot(vo_inter[:,0], vo_inter[:,1], marker='.', color='b', label="visual odometry")
plt.plot(vo_inter[::5,0], vo_inter[::5,1], marker='.', color='k', ls='')
#for i in range(0,len(vo_inter),10):
#  plt.text(vo_inter[i,0]+2, vo_inter[i,1]+2, str(i), color='b')
#  plt.text(gps_pts[i,0]+2, gps_pts[i,1]+2, str(i), color='r')
plt.xlabel("x (m)", fontsize=26)
plt.ylabel("z (m)", fontsize=26)
plt.legend(loc="upper left", fontsize=22)

#fig_path.savefig("plot_path_diff.png", bbox_inches='tight', dpi=200)
fig_path.savefig("plot_path_diff.pdf", bbox_inches='tight')
fig_speed.savefig("plot_speed.pdf", bbox_inches='tight')
fig_scale.savefig('plot_scale_error.pdf', bbox_inches='tight')
fig_rot.savefig("plot_rotation_diff.pdf", bbox_inches='tight')

plt.show()
exit(0)

# save demo images
#fig_demo =  plt.figure(figsize=(10,10), dpi=110)
#plt.axis([-50, 200, -100, 150], 'equal')
#plt.axis('off')
#
#t = vo_start
#gps_cnt = 0
#plt.plot(vo_pts2D[0,0], vo_pts2D[0,1], color='b')
#fig_demo.savefig("demo_plot/img_plot_%06d.png" % (0), bbox_inches='tight', dpi=110)
#plt.clf()
#for i in range(vo_pts2D.shape[0]):
#   plt.axis([-50, 200, -100, 150], 'equal')
#   plt.axis('off')
#
#   if vo_times[i] >= t:
#      t += 1.0
#      gps_cnt += 1
#
#   #print(vo_inter[0:gps_cnt,:])
#   #print(vo_pts2D[0:i+1,:])
#   plt.plot(vo_pts_crop[0:i+1,0], vo_pts_crop[0:i+1,1], marker=" ", color='b')
#   plt.plot(vo_inter[0:gps_cnt,0], vo_inter[0:gps_cnt,1], marker='.', color='b')
#   plt.plot(vo_inter[0:gps_cnt:5,0], vo_inter[0:gps_cnt:5,1], marker='.', color='k', ls='')
#
#   if i >= 927:
#      gps_pts = np.vstack((gps_pts, gps_pts[0,:]))
#      plt.plot(gps_pts[:,0], gps_pts[:,1], marker='.', color='r')
#   plt.plot(gps_pts[0:gps_cnt,0], gps_pts[0:gps_cnt,1], marker='.', color='r')
#   plt.plot(gps_pts[0:gps_cnt:5,0], gps_pts[0:gps_cnt:5,1], marker='.', color='k', ls="")
#
#   print(i)
#   #fig_demo.savefig("demo_plot/img_plot_%06d.png" % (i+1), bbox_inches='tight', dpi=200)
#   fig_demo.savefig("demo_plot/img_plot_%06d.png" % (i+1), bbox_inches='tight', dpi=110)
#   plt.clf()



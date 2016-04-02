#!/bin/python

import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import np_helper  as nph

# generate 3D points around points in pts_center using gaussian distribution
def generate_points(Rt_inv, pts_center, pts_sigma, pts_fixed, pts_num, axis_range):
   # allocate mem for points
   points = np.zeros([4, pts_num.sum()])
   istart = 0
   for i in range(pts_center.shape[1]):
      #print(pts_center[:,i])
      for j in range(pts_num[i]):
         for k in range(3):
            #print(i,j,k)
            if pts_fixed[k,i] == 0:
               points[k,istart+j] = np.random.normal(pts_center[k,i], pts_sigma[k,i])
            else:
               points[k,istart+j] = pts_center[k,i]
            # force axis range if outside of domain
            if points[k,istart+j] < axis_range[k,0]:
               points[k,istart+j] = axis_range[k,0]
            elif points[k,istart+j] > axis_range[k,1]:
               points[k,istart+j] = axis_range[k,1]

         points[3,istart+j] = 1.0
         # transform in current camera position 
         points[:,istart+j] = Rt_inv.dot(points[:,istart+j])

      istart += pts_num[i]
   #print(points)
   return points

# plot 3d points
def plot_pts3d(pts3d, visible_status):
   fig_xz = plt.figure()
   plt.xlabel('X (m)', fontsize=14)
   plt.ylabel('Z (m)', fontsize=14)
   plt.axis('equal')
   plt.plot(pts3d[0,:], pts3d[2,:], "ro")
   for i in range(visible_status.shape[0]):
      if visible_status[i] == 0:
         plt.plot(pts3d[0,i], pts3d[2,i], "ko")
   fig_xy = plt.figure()
   plt.xlabel('X (m)', fontsize=14)
   plt.ylabel('Y (m)', fontsize=14)
   plt.axis('equal')
   plt.gca().invert_yaxis()   
   plt.plot(pts3d[0,:], pts3d[1,:], 'bo')
   for i in range(visible_status.shape[0]):
      if visible_status[i] == 0:
         plt.plot(pts3d[0,i], pts3d[1,i], "ko")

   fig_xz.savefig("plot_pts3d_xz.pdf", bbox_inches='tight')
   fig_xy.savefig("plot_pts3d_xy.pdf", bbox_inches='tight')
   # non blocking
   #plt.ion()
   # or
   #plt.draw()
   #plt.show()

# plot optical flow
def plot_flow(pts2d_left, pts2d_right, imgsz):
   fig_flow = plt.figure()
   #plt.xlim(0, width)
   #plt.ylim(0, height)
   plt.axis([0, imgsz[0], 0, imgsz[1]], 'equal')
   plt.gca().invert_yaxis()
   plt.xlabel('u (pixels)', fontsize=14)
   plt.ylabel('v (pixels)', fontsize=14)

   for i in range(pts2d_left.shape[1]):
      # if not visible in cam - skip it
      if pts2d_left[0,i] < -0.5 or pts2d_right[0,i] < -0.5:
         continue
      match_x = np.array([pts2d_left[0,i], pts2d_right[0,i]])
      match_y = np.array([pts2d_left[1,i], pts2d_right[1,i]])
      plt.plot(match_x, match_y, 'k.-')
      plt.plot(pts2d_left[0,i], pts2d_left[1,i], 'r.')      
      plt.plot(pts2d_right[0,i], pts2d_right[1,i], 'b.')

   fig_flow.savefig("plot_artif_disp.pdf", bbox_inches='tight')   

# project 3d points onto image plane
def project_points(imgsz, C, Rt, pts3d):
   pts2d = C.dot(Rt.dot(pts3d))
   pts2d = pts2d / pts2d[2,:]
   return pts2d[0:2,:]


# convert projected 2d points into pixels - simulates camera sensor
def pixelize(imgsz, pts2d):
   for i in range(pts2d.shape[1]):
      # if in sensor range - pixelize it
      #print("pt ", i, "\n", pts2d[:,i])
      #if pts2d[0,i] >= 0 and pts2d[0,i] <= width and pts2d[1,i] >= 0 and pts2d[1,i] <= height:
      if pts2d[0,i] >= -0.5 and pts2d[0,i] <= (imgsz[0]-0.5) and pts2d[1,i] >= -0.5 and pts2d[1,i] <= (imgsz[1]-0.5):
         pts2d[:,i] = np.round(pts2d[:,i])
      # else remove that point
      else:
         pts2d[:,i] = -1


def getErrors(pts3d, projs, C, Rt_prev, Rt_curr, baseline, imgsz, is_pixelized=True):
   f = C[0,0]
   cu = C[0,2]
   cv = C[1,2]
   pts3d_prev = Rt_prev.dot(pts3d)
   #pts3d_curr = Rt_curr.dot(pts3d)
   # Rt_prev * Rt_inc = Rt_curr
   #Rt_inc = nph.inv_Rt(Rt_prev).dot(Rt_curr)
   # Rt_curr * Rt_inc = Rt_prev
   #Rt_inc = nph.inv_Rt(Rt_curr).dot(Rt_prev)
   Rt_inc = nph.inv_Rt(Rt_prev.dot(nph.inv_Rt(Rt_curr)))
   triangpt_p = np.zeros((4))
   triang_err = reproj_err = 0.0
   num_pts = pts3d.shape[1]
   for i in range(num_pts):
      if projs[0,0,i] < -0.5 or projs[1,0,i] < -0.5 or projs[2,0,i] < -0.5 or projs[3,0,i] < -0.5:
         num_pts -= 1
         continue
      # triangulate in previous left camera
      assert (projs[0,0,i] - projs[1,0,i]) >= 0
      if is_pixelized:
         d = np.max([projs[0,0,i] - projs[1,0,i], 0.001])
      else:
         d = projs[0,0,i] - projs[1,0,i]
      assert d > 0.0
      triangpt_p[0] = (projs[0,0,i] - cu) * baseline / d
      triangpt_p[1] = (projs[0,1,i] - cv) * baseline / d
      triangpt_p[2] = f * baseline / d
      triangpt_p[3] = 1.0
      # calculate triangulation error
      triang_err += np.linalg.norm(pts3d_prev[:,i] - triangpt_p)
      # calculate reprojection error
      # TODO
      # project in prev left frame - same as projs[0,:,i]
      ptproj_p = project_points(imgsz, C, np.eye(4), triangpt_p.reshape(4,1))
      # project in current left frame
      ptproj_c = project_points(imgsz, C, Rt_inc, triangpt_p.reshape(4,1))
      pt2d_lc = projs[2,:,i].reshape(2,1)
      reproj_err += np.linalg.norm(pt2d_lc - ptproj_c)      

   assert num_pts > 0
   return [triang_err/num_pts, reproj_err/num_pts, num_pts]



def write_points(pts3d, projs, filename):
   fp = open(filename, "w")
   for i in range(pts3d.shape[1]):
      # if point not visible in some of 4 images, skip
      if projs[0,0,i] < -0.5 or projs[1,0,i] < -0.5 or projs[2,0,i] < -0.5 or projs[3,0,i] < -0.5:
         continue
      fp.write(str(pts3d[0,i]) + " " + str(pts3d[1,i]) + " " + str(pts3d[2,i]))
      fp.write(" 2")
      # write left and right features for every frame
      for j in range(2):
         fp.write(" " + str(j) + " " + str(projs[0+(j*2),0,i]) + " " + str(projs[0+(j*2),1,i]) + " " + 
                  str(projs[1+(j*2),0,i]) + " " + str(projs[1+(j*2),1,i]))
      fp.write("\n")
   fp.close()

def getVisibleStatus(projs):
   status = np.ones(projs.shape[2], dtype=np.int8)
   for i in range(status.shape[0]):
      if projs[0,0,i] < -0.5 or projs[1,0,i] < -0.5 or projs[2,0,i] < -0.5 or projs[3,0,i] < -0.5:
         status[i] = 0
   return status

def generate_test_scene():
   pts3d = np.array([[-2, -2, 10, 1], [2, -2, 10, 1], [0, 2, 10, 1], [-0.5, -0.5, 2, 1]]).T
   return pts3d

# main
np.set_printoptions(precision=3, linewidth=180)

#campose_path = "/home/kreso/projects/master_thesis/datasets/libviso/odometry_pose/poses/00.txt"
#campose_path = "/home/kreso/projects/master_thesis/datasets/libviso/odometry_pose/croped/00_crop.txt"
#campose_path = "/home/kreso/projects/master_thesis/datasets/libviso/odometry_pose/croped/00_crop.txt"
path_file = "/home/kreso/projects/datasets/KITTI/odometry_pose/croped/00_crop.txt"
#path_file = "path_170m.txt"
# bumblebee
#imgsz = np.array([640, 480])
#cam_mat = "C_bb.txt"
#base = 0.12
#out_folder_prefix = "/home/kreso/projects/master_thesis/datasets/stereo_model/points_170m_bbcam_nonoise_base_"

# libviso KITTI cam
imgsz = np.array([1241,376])
cam_mat = "C_libviso_00.txt"
out_folder_prefix = "/home/kreso/projects/datasets/stereo_model//points_kitti_cam_nonoise_base_"

# axis domains
range_x = [-40, 40]
range_y = [-20, 8] # -20, 3
range_z = [5, 150]

# best
#range_x = [-30, 30]
#range_y = [-20, 8] # -20, 3
#range_z = [10, 150]
axis_range = np.array([range_x, range_y, range_z])
print(axis_range)
# point centroids
# close
#pts_center = np.array([[-5, -2, 20, 1], [5, -2, 20, 1], [0, -2, 20, 1]]).T
# far
#pts_center = np.array([[-10, -5, 60, 1], [10, -5, 60, 1], [0, -5, 80, 1]]).T
# TODO prosorit tocke da ima koja i u blizini
#pts_center = np.array([[-15, -2, 80, 1], [15, -2, 80, 1], [0, -2, 80, 1]]).T
# best
pts_center = np.array([[-10, -2, 60, 1], [10, -2, 60, 1], [0, -2, 90, 1]]).T
# coords with fixed values are marked with 1
#pts_fixed = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]]).transpose()
pts_fixed = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).T
# max and min range of points coords
#pts_range = np.array([range_x, range_y, range_z])
# sigma value for gauss distribution
# close
#pts_sigma = np.array([[1, 4, 6], [1, 4, 6], [4, 4, 8]]).T
# far
#pts_sigma = np.array([[2, 4, 20], [2, 4, 20], [4, 4, 20]]).T
#pts_sigma = np.array([[10, 4, 20], [10, 4, 20], [10, 4, 20]]).T
# best
pts_sigma = np.array([[6, 6, 20], [6, 6, 20], [10, 6, 20]]).T

# number of points per center
# many
#pts_num = np.array([50, 50, 100]) #30 30 30
# few - bb cam libviso - 50/60 inliers
pts_num = np.array([60, 60, 80])

#print(pts_center, "\n", pts_range, "\n", pts_sigma, pts_fixed, pts_num)
Rt_I = np.eye(4)

#C = np.eye(3)
#np.savetxt('C.txt', C, fmt='%.2f')
C = np.loadtxt(cam_mat)
print('C:\n', C, '\n')

Rt_mats = np.loadtxt(path_file)
Rt_mats = np.append(Rt_mats, np.zeros((Rt_mats.shape[0], 3)), 1)          # add three zero columns
Rt_mats = np.append(Rt_mats, np.array([[1] * Rt_mats.shape[0]]).T, 1)   # add one ones column
print("Rt mats: \n", Rt_mats, "\n")


nframes = Rt_mats.shape[0]
projs = np.zeros((4, 2, pts_num.sum()))
pts3d = np.zeros((nframes, 4, pts_num.sum()))


for i in range(nframes-1):
   # inputs are camera position matrices in each frame
   # so they are inverse of points transform Rt matrix
   Rt_prev_inv = Rt_mats[i,:].reshape(4,4)
   Rt_curr_inv = Rt_mats[i+1,:].reshape(4,4)
   #projs = np.zeros((4, 2, 4))
   #pts3d = generate_test_scene()
   # generate new 3D points in front of current camera position
   #pts3d = generate_points(Rt_prev_inv.dot(pts_center), pts_sigma, pts_fixed, pts_num, axis_range)
   pts3d[i,:,:] = generate_points(Rt_prev_inv, pts_center, pts_sigma, pts_fixed, pts_num, axis_range)


baseline = np.array([0.53716])
#baseline = np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0])
triang_errors = np.zeros((baseline.shape[0], nframes-1))
for b in range(baseline.shape[0]):
   out_folder = out_folder_prefix + "%.2f/" % baseline[b]
   print("output folder -> ", out_folder)
   if os.path.exists(out_folder):
      shutil.rmtree(out_folder)
      #os.rmdir(out_folder)
   os.makedirs(out_folder)

   # Tb transform puts right camera in center
   Tb = np.eye(4)
   Tb[0,3] = -baseline[b]
   print("Tb:\n", Tb, "\n")
   for i in range(nframes-1):
      # inputs are camera position matrices in each frame
      # so they are inverse of points transform Rt matrix
      Rt_prev_inv = Rt_mats[i,:].reshape(4,4)
      Rt_curr_inv = Rt_mats[i+1,:].reshape(4,4)
      
      # calculate point trasform Rt matrices in 2 frames (inverse of camera transform matrices)
      # slower way
      # faster (and better) way
      Rt_prev = nph.inv_Rt(Rt_prev_inv)
      Rt_curr =  nph.inv_Rt(Rt_curr_inv)
      #print(Rt_prev)
      Rt_prev = nph.inv_Rt(Rt_prev_inv)
      Rt_curr =  nph.inv_Rt(Rt_curr_inv)

      # project 3d point on image plane
      pts2d_leftp = project_points(imgsz, C, Rt_prev, pts3d[i,:,:])
      pts2d_rightp = project_points(imgsz, C, Tb.dot(Rt_prev), pts3d[i,:,:])
      # round them up in pixels
      # do the same for current frame
      pts2d_leftc = project_points(imgsz, C, Rt_curr, pts3d[i,:,:])
      pts2d_rightc = project_points(imgsz, C, Tb.dot(Rt_curr), pts3d[i,:,:])

      projs[0,:,:] = pts2d_leftp
      projs[1,:,:] = pts2d_rightp
      projs[2,:,:] = pts2d_leftc
      projs[3,:,:] = pts2d_rightc
      [triang_err, reproj_err, visible_num] = getErrors(pts3d[i,:,:], projs, C, Rt_prev, Rt_curr, baseline[b], imgsz, False)
      print("Frame " + str(i) + "\npoints visible: " + "%d / %d" % (visible_num, pts3d.shape[2]))
      print("reproj error: " + str(reproj_err) + "\ntriangulation error: " + "%.4f" % triang_err + "\n\n")
      triang_errors[b,i] = triang_err
      
      # write 3d points and 2d projs in files
      write_points(pts3d[i,:,:], projs, out_folder + "point_projs_" + "%06d" % (i+1) + ".txt") 

      #print(pts2d_leftc.transpose())
      #print(pts2d_rightc.transpose())
      
      #visible_pts = getVisibleStatus(projs)
      #plot_pts3d(pts3d[i,:,:], visible_pts)
      #plot_flow(pts2d_leftc, pts2d_rightc, imgsz)
      #plt.show()
      #exit(0)

#fig_triang = plt.figure()
#plt.axis([0, 185, 0, 35], 'equal')
#plt.plot(triang_errors[0,:], "r-", label="Bumblebee cam")
#plt.plot(triang_errors[1,:], "b-", label="KITTI cam")
#plt.xlabel('frame number', fontsize=30)
#plt.ylabel('triangulation error (m)', fontsize=30)
#plt.legend(fontsize=22)
#plt.show()
#fig_triang.savefig("plot_triang_error.pdf", bbox_inches='tight')



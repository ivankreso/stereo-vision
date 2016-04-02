#!/bin/python
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import np_helper  as nph


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
    for i in range(pts2d.shape[1]):
        if pts2d[2,i] > 1.0:
            pts2d[:,i] = pts2d[:,i] / pts2d[2,i]
        else:
            pts2d[:,i] = -1.0
            #print(pts2d[:,i])
    return pts2d[0:2,:]

# convert projected 2d points into pixels - simulates camera sensor
def pixelize(imgsz, pts2d):
    for i in range(pts2d.shape[1]):
        # if in sensor range - pixelize it
        if pts2d[0,i] >= -0.5 and pts2d[0,i] <= (imgsz[0]-0.5) and pts2d[1,i] >= -0.5 and pts2d[1,i] <= (imgsz[1]-0.5):
            #continue
            pts2d[:,i] = np.round(pts2d[:,i]) # SBA slightly better with SWS = 5
            # add gaussian noise
            #noise = np.random.normal(0.0, 0.2)
            #noise = np.random.normal(0.0, 0.3) # SBA still better
            #noise = np.random.normal(0.0, 0.4) # worse
            #pts2d[:,i] = pts2d[:,i] + noise
        # else remove that point
        else:
            pts2d[:,i] = -1.0

def getVisibleStatus(projs):
   status = np.ones(projs.shape[2], dtype=np.int8)
   for i in range(status.shape[0]):
      if projs[0,0,i] < -0.5 or projs[1,0,i] < -0.5 or projs[2,0,i] < -0.5 or projs[3,0,i] < -0.5:
         status[i] = 0
   return status

def triangulate(C, b, proj_left, projs_right):
    f = C[0,0]
    cx = C[0,2]
    cy = C[1,2]
    x = proj_left[0]
    y = proj_left[1]
    pt3d = np.zeros((3))
    disp = x - projs_right[0]

    disp = max(disp, 0.001)
    pt3d[0] = (x - cx) * b / disp;
    pt3d[1] = (y - cy) * b / disp;
    pt3d[2] = f * b / disp;
    return pt3d

def update_age(age, projs_left, projs_right, frame):
    if frame == 0:
        for i in range(age.shape[1]):
            age[frame,i] = 0
    else:
        for i in range(age.shape[1]):
            if projs_left[0,i] < -0.5 or projs_left[0,i] < -0.5 or projs_right[0,i] < -0.5 or projs_right[0,i] < -0.5:
                age[frame,i] = -1
            else:
                age[frame,i] = age[frame-1,i] + 1

def write_tracker_data(folder, projs_left, projs_right, age):
    num_frames = projs_left.shape[0]
    # write 3d points and 2d projs in files
    for i in range(num_frames):
        write_frame_projs(i, folder + "/%06d" % (i) + ".txt", projs_left[i,:,:], projs_right[i,:,:], age[i,:])


def write_frame_projs(i, filename, projs_left, projs_right, age):
    fp = open(filename, "w")
    for i in range(projs_left.shape[1]):
        # if point not visible in some of 4 images, skip
        fp.write(str(i) + " " + str(age[i]))
        # write left and right features for every frame
        fp.write(" " + str(projs_left[0,i]) + " " + str(projs_left[1,i]) + " "
                 + str(projs_right[0,i]) + " " + str(projs_right[1,i]))
        fp.write("\n")
    fp.close()

def write_points_sba(filename, C, baseline, extr_params, projs_left, projs_right, pts3d_gt):
    num_world_pts = projs_left.shape[2]
    fp = open(filename, "w")

    fp.write(str(C[0,0]) + " " + str(C[1,1]) + " " + str(C[0,2]) + " " + str(C[1,2]) + " " + str(baseline) + "\n")

    pts3d_lst = []
    observ_left_lst = []
    observ_right_lst = []
    #projs_left = np.zeros((nframes, 2, pts3d_num))
    #points3d = np.array(3, num_world_pts)
    assert projs_left.shape[0] == extr_params.shape[0]
    num_frames = projs_left.shape[0]
    num_points = 0
    num_observations = 0
    for i in range(num_world_pts):
        # if visible in first frame add that point and all its observations
        if projs_left[0,0,i] >= 0.0 and projs_right[0,0,i] >= 0.0:
            num_points += 1
            #points3d[:,i] = triangulate(C, baseline, projs_left[0,:,i], projs_right[0,:,i])
            pts3d_lst.append(triangulate(C, baseline, projs_left[0,:,i], projs_right[0,:,i]))
            print(pts3d_lst[-1].T, " --> ", pts3d_gt[:,i])

            observ_left = np.ndarray(shape=(2,0))
            observ_right = np.ndarray(shape=(2,0))
            for f in range(num_frames):
                # add until we find unvisible projection
                if projs_left[f,0,i] >= 0.0 and projs_right[f,0,i] >= 0.0:
                    #print(projs_left[f,:,i].reshape(2,1))
                    observ_left = np.hstack([observ_left, projs_left[f,:,i].reshape(2,1)])
                    observ_right = np.hstack([observ_right, projs_right[f,:,i].reshape(2,1)])
                    num_observations += 1
                else:
                    break
            observ_left_lst.append(observ_left)
            observ_right_lst.append(observ_right)
    #pts3d = np.array(pts3d_lst)
    fp.write(str(num_frames) + " " + str(num_points) + " " + str(num_observations) + "\n")
    for i in range(len(observ_left_lst)):
        left = observ_left_lst[i]
        right = observ_right_lst[i]
        for f in range(left.shape[1]):
            fp.write(str(f) + " " + str(i) + " " + str(left[0,f]) + " " + str(left[1,f]) + " "
                     + str(right[0,f]) + " " + str(right[1,f]) + "\n")

    for i in range(extr_params.shape[0]):
        #R = Rt[i,:].reshape(4,4)[0:4,0:4]
        #(rvec, jac) = cv2.Rodrigues(R)
        rt_vec = extr_params[i,:]
        for j in range(rt_vec.shape[0]):
            fp.write(str(rt_vec[j]) + "\n")
    for i in range(len(pts3d_lst)):
        pts3d = pts3d_lst[i]
        print(pts3d)
        for j in range(3):
            fp.write(str(pts3d[j]) + "\n")

# main
np.set_printoptions(precision=3, linewidth=180)
path_file = "path.txt"
path_estim = "path_noise.txt"
out_folder = "/home/kivan/Projects/datasets/stereo_sba/"

# bumblebee
#imgsz = np.array([640, 480])
#cam_mat = "C_bb.txt"
#baseline = 0.12
#out_folder_prefix = "/home/kivan/Projects/datasets/stereo_sba/"

# libviso 00 cam
imgsz = np.array([1241,376])
cam_mat = "C_libviso_00.txt"
baseline = 0.53716
#out_folder = "/home/kreso/projects/master_thesis/datasets/stereo_model/pointdata_viso00path_00cam/"

Rt_I = np.eye(4)
#C = np.eye(3)
#np.savetxt('C.txt', C, fmt='%.2f')
C = np.loadtxt(cam_mat)
print('C:\n', C, '\n')

extr_noise = np.loadtxt(path_estim)

Rt_mats = np.loadtxt(path_file)
Rt_mats = np.append(Rt_mats, np.zeros((Rt_mats.shape[0], 3)), 1)          # add three zero columns
Rt_mats = np.append(Rt_mats, np.array([[1] * Rt_mats.shape[0]]).T, 1)   # add one ones column
#print("Rt mats: \n", Rt_mats, "\n")

nframes = Rt_mats.shape[0]

# generate new 3D points in front of current camera position
pts3d = np.loadtxt("pts3d.txt")
#print(pts3d)

pts3d_num = pts3d.shape[1]
projs_left = np.zeros((nframes, 2, pts3d_num))
projs_right = np.zeros((nframes, 2, pts3d_num))
age = np.zeros((nframes, pts3d_num), dtype='int32')

# Tb transform puts right camera in center
Tb = np.eye(4)
Tb[0,3] = - baseline
print("Tb:\n", Tb, "\n")
for i in range(nframes):
    ## inputs are camera position matrices in each frame
    ## so they are inverse of points transform Rt matrix
    #Rt_prev_inv = Rt_mats[i,:].reshape(4,4)
    #Rt_curr_inv = Rt_mats[i+1,:].reshape(4,4)
    ## calculate point trasform Rt matrices in 2 frames (inverse of camera transform matrices)
    ## slower way
    ##Rt_prev = np.linalg.inv(Rt_prev_inv)
    ##Rt_curr =  np.linalg.inv(Rt_curr_inv)
    ## faster (and better) way
    #Rt_prev = nph.inv_Rt(Rt_prev_inv)
    #Rt_curr =  nph.inv_Rt(Rt_curr_inv)

    #print(Rt_prev)
    #print(nph.inv_Rt(Rt_prev_inv))
    # project 3d point on image plane
    print("Frame: " + str(i))
    Rt = nph.inv_Rt(Rt_mats[i,:].reshape(4,4))
    pts2d_left = project_points(imgsz, C, Rt, pts3d)
    pts2d_right = project_points(imgsz, C, Tb.dot(Rt), pts3d)
    # round them up in pixels
    pixelize(imgsz, pts2d_left)
    pixelize(imgsz, pts2d_right)
    update_age(age, pts2d_left, pts2d_right, i)

    projs_left[i,:,:] = pts2d_left
    projs_right[i,:,:] = pts2d_right
    #print("Frame " + str(i) + "\npoints visible: " + "%d / %d" % (visible_num, pts3d.shape[2]))

    #print("Plotting 3d points")
    #visible_pts = getVisibleStatus(projs_left)
    #plot_pts3d(pts3d, visible_pts)
    #plot_flow(pts2d_left, pts2d_right, imgsz)
    #plt.show()
    #exit(0)

# TODO: remove pts3d - write_points_sba("SBA_dataset.txt", C, baseline, extr_noise, projs_left, projs_right, pts3d)
write_tracker_data(out_folder, projs_left, projs_right, age)

#fig_triang = plt.figure()
#plt.axis([0, 185, 0, 35], 'equal')
#plt.plot(triang_errors[0,:], "r-", label="Bumblebee cam")
#plt.plot(triang_errors[1,:], "b-", label="KITTI cam")
#plt.xlabel('frame number', fontsize=30)
#plt.ylabel('triangulation error (m)', fontsize=30)
#plt.legend(fontsize=22)
#plt.show()
#fig_triang.savefig("plot_triang_error.pdf", bbox_inches='tight')

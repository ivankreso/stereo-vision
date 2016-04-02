#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt


#path_dir1 = "/home/kivan/Projects/cv-stereo/experiments/kitti/results/tracker_freak_ba_7_ba/"
#path_dir2 = "/home/kivan/Projects/cv-stereo/experiments/kitti/results/tracker_freak_ba_7/"
#path_dir1 = "/home/kivan/Projects/cv-stereo/build/vo_batch/release/tsukuba/tracker_ncc_tsukuba_4/"
#path_dir2 = "/home/kivan/Projects/cv-stereo/build/vo_batch/release/tsukuba/tracker_ncc_tsukuba_4_ba/"
#path_dir1 = "/home/kivan/Projects/cv-stereo/build/vo_batch_workstation/release/tsukuba/tracker_ncc_tsukuba_3/"
#path_dir2 = "/home/kivan/Projects/cv-stereo/build/vo_batch_workstation/release/tsukuba/tracker_ncc_tsukuba_3_ba/"
path_dir1 = "/home/kivan/Projects/cv-stereo/experiments/kitti/results/tracker_freak_ba_7_2_ba"
path_dir2 = "/home/kivan/Projects/cv-stereo/experiments/kitti/results/tracker_freak_ba_7_2"

def get_error_diff(path_dir1, path_dir2, i):
    filepath1 = path_dir1 + "/" + ("errors_%02d.txt" % i)
    filepath2 = path_dir2 + "/" + ("errors_%02d.txt" % i)
    error_data1 = np.array(np.loadtxt(filepath1))
    error_data2 = np.array(np.loadtxt(filepath2))
    print(error_data1.shape)
    error_data1 = error_data1[7::8]
    error_data2 = error_data2[7::8]

    fig_scale = plt.figure(figsize=(12,8))
    plt.plot(error_data1[:,2] - error_data2[:,2], color='r', label="translation")
    plt.plot(error_data1[:,1] - error_data2[:,1], color='b', label="rotation")
    plt.legend(fontsize=24)


def get_mean_var(path_dir):
    #for i in range(11):
    for i in range(1):
        filepath = path_dir + "/" + ("errors_%02d.txt" % i)
        print("\n", filepath)
        error_data = np.array(np.loadtxt(filepath))
        seq_trans_mean = 0.0
        seq_trans_var = 0.0
        seq_rot_mean = 0.0
        seq_rot_var = 0.0
        # calc mean and variance
        N = error_data.shape[0]
        for j in range(N):
            trans_err = error_data[j,2]
            rot_err = error_data[j,1]
            seq_trans_mean += trans_err
            seq_rot_mean += rot_err
            seq_trans_var += trans_err**2
            seq_rot_var += rot_err**2

        seq_trans_mean = seq_trans_mean / N
        seq_rot_mean = seq_rot_mean / N
        seq_trans_var = (seq_trans_var / N) - seq_trans_mean**2
        seq_rot_var = (seq_rot_var / N) - seq_rot_mean**2
        print("Translation:\nMean = %e\nVariance = %e" % (seq_trans_mean, seq_trans_var))
        print("Rotation:\nMean = %e\nVariance = %e" % (seq_rot_mean, seq_rot_var))

        fig_scale = plt.figure(figsize=(12,8))
        plt.plot(error_data[:,2], color='r', label="translation")
        plt.plot(error_data[:,1], color='b', label="rotation")
        plt.legend(fontsize=24)

        return [seq_trans_mean, seq_trans_var, seq_rot_mean, seq_rot_var]


#(mean1, var1, mean2, var2) = get_mean_var(path_dir1)
#(mean1, var1, mean2, var2) = get_mean_var(path_dir2)
#get_error_diff(path_dir1, path_dir2, 1)

#filepath = "/home/kivan/Projects/cv-stereo/stereo_odometry/evaluation/errors_gt_wgtoff.txt"
filepath = "/home/kivan/Projects/cv-stereo/stereo_odometry/evaluation/errors.txt"
#filepath = "/home/kivan/Projects/cv-stereo/stereo_odometry/evaluation/errors_vo_wgton.txt"
error_data = np.array(np.loadtxt(filepath))
fig_scale = plt.figure(figsize=(12,8))
plt.plot(error_data[:,2], color='r', label="translation")
plt.plot(error_data[:,1], color='b', label="rotation")
plt.legend(fontsize=24)
plt.title(filepath)

plt.show()

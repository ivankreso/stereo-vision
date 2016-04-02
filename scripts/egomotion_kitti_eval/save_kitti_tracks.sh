#!/bin/bash

#experiment_config="../../../config_files/experiments/kitti/tracker_ncc_save.txt"
#experiment_config="../../../config_files/experiments/kitti/ncc_save_uniform.txt"
#experiment_config="../../../config_files/experiments/kitti/ncc_save_best.txt"

# the best so far
experiment_config="../../../config_files/experiments/kitti/train_df/ncc_train_df.txt"

#cd /home/kivan/Projects/cv-stereo/build/save_tracks_without_ransac/release/
cd /home/kivan/source/cv-stereo/build/save_tracks/release/
#save_folder="/mnt/ssd/kivan/datasets/tracker_data/ncc/"
save_folder="/mnt/ssd/kivan/datasets/tracker_data/ncc/"

#cd /home/kivan/Projects/cv-stereo/build/save_tracks/release/
#save_folder="/mnt/ssd/kivan/datasets/tracker_data/ncc_with_ransac/"

#save_folder="/opt/kivan/datasets/tracker_data/ncc/"

#OMP_NUM_THREADS=22 ./save_tracks -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_00.txt -e $experiment_config -s $save_folder
#OMP_NUM_THREADS=22 ./save_tracks -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_01.txt -e $experiment_config -s $save_folder
#OMP_NUM_THREADS=22 ./save_tracks -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_02.txt -e $experiment_config -s $save_folder

OMP_NUM_THREADS=4 ./save_tracks -c /home/kivan/source/cv-stereo/config_files/config_kitti_04.txt -e $experiment_config -s $save_folder
OMP_NUM_THREADS=4 ./save_tracks -c /home/kivan/source/cv-stereo/config_files/config_kitti_05.txt -e $experiment_config -s $save_folder
OMP_NUM_THREADS=4 ./save_tracks -c /home/kivan/source/cv-stereo/config_files/config_kitti_06.txt -e $experiment_config -s $save_folder
OMP_NUM_THREADS=4 ./save_tracks -c /home/kivan/source/cv-stereo/config_files/config_kitti_07.txt -e $experiment_config -s $save_folder
OMP_NUM_THREADS=4 ./save_tracks -c /home/kivan/source/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -s $save_folder
OMP_NUM_THREADS=4 ./save_tracks -c /home/kivan/source/cv-stereo/config_files/config_kitti_09.txt -e $experiment_config -s $save_folder
OMP_NUM_THREADS=4 ./save_tracks -c /home/kivan/source/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -s $save_folder


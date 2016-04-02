#!/bin/bash

#bin_folder="/home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/"
bin_folder="/home/kivan/Dropbox/binaries/vo_batch_debug/"

#./run_kitti_evaluation_all.sh "../../../config_files/experiments/tracker_freak_ba_7_3.txt" $bin_folder

#./run_kitti_evaluation_test.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tracker_freak_7_1.txt" $bin_folder

#./run_kitti_evaluation_elbereth.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tracker_freak_7_6_1.txt" $bin_folder
#./run_kitti_evaluation_elbereth.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tracker_freak_7_6_2.txt" $bin_folder
#./run_kitti_evaluation_elbereth.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tracker_freak_7_6_3.txt" $bin_folder
#./run_kitti_evaluation_elbereth.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tracker_freak_7_6_4.txt" $bin_folder
#./run_kitti_evaluation_elbereth.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tracker_freak_7_6_5.txt" $bin_folder
#./run_kitti_evaluation_elbereth.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tracker_freak_7_6_6.txt" $bin_folder

./run_kitti_evaluation_elbereth.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tracker_ncc_ccvw14_3.txt" $bin_folder
./run_kitti_evaluation_elbereth.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tracker_ncc_ccvw14_4.txt" $bin_folder

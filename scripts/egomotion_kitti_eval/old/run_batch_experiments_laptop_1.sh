#!/bin/bash

bin_folder="/home/kivan/Projects/cv-stereo/build/vo_batch_debug/release2/"

#./run_tsukuba_evaluation_elbereth.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tsukuba/ccvw14_nowgt_tracker_freak_tsukuba_1.txt" $bin_folder
#./run_tsukuba_evaluation_elbereth.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tsukuba/ccvw14_nowgt_tracker_freak_tsukuba_2.txt" $bin_folder
#./run_tsukuba_evaluation_elbereth.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tsukuba/ccvw14_nowgt_tracker_freak_tsukuba_3.txt" $bin_folder
#./run_tsukuba_evaluation_elbereth.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/tsukuba/ccvw14_nowgt_tracker_freak_tsukuba_4.txt" $bin_folder

./run_kitti_evaluation_all.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/ccvw14_nowgt_tracker_ncc_1.txt" $bin_folder
./run_kitti_evaluation_all.sh "/home/kivan/Projects/cv-stereo/config_files/experiments/ccvw14_nowgt_tracker_ncc_2.txt" $bin_folder

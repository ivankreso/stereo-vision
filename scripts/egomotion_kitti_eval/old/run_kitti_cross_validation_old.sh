#!/bin/bash

cd /home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/

experiment_config_df="../../../config_files/experiments/kitti/tracker_ncc_validate_df.txt"
experiment_config_nodf="../../../config_files/experiments/kitti/tracker_ncc_validate_nodf.txt"

#./visodom -c ../../../config_files/config_kitti_00.txt -e $experiment_config
#./visodom -c ../../../config_files/config_kitti_01.txt -e $experiment_config
#./visodom -c ../../../config_files/config_kitti_02.txt -e $experiment_config
#./visodom -c ../../../config_files/config_kitti_03.txt -e $experiment_config
#./visodom -c ../../../config_files/config_kitti_04.txt -e $experiment_config
#./visodom -c ../../../config_files/config_kitti_05.txt -e $experiment_config
#./visodom -c ../../../config_files/config_kitti_06.txt -e $experiment_config
#./visodom -c ../../../config_files/config_kitti_07.txt -e $experiment_config
#./visodom -c ../../../config_files/config_kitti_08.txt -e $experiment_config
#./visodom -c ../../../config_files/config_kitti_09.txt -e $experiment_config
#./visodom -c ../../../config_files/config_kitti_10.txt -e $experiment_config

#./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_00.txt -e $experiment_config
#./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_01.txt -e $experiment_config
#./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_02.txt -e $experiment_config
#./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_03.txt -e $experiment_config

./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_04.txt -e $experiment_config_df
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_04.txt -e $experiment_config_nodf
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_05.txt -e $experiment_config_df
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_05.txt -e $experiment_config_nodf
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_06.txt -e $experiment_config_df
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_06.txt -e $experiment_config_nodf
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_07.txt -e $experiment_config_df
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_07.txt -e $experiment_config_nodf
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_08.txt -e $experiment_config_df
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_08.txt -e $experiment_config_nodf
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_09.txt -e $experiment_config_df
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_09.txt -e $experiment_config_nodf
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_10.txt -e $experiment_config_df
./visodom -c /home/kivan/Projects/cv-stereo/config_files/validation_config_kitti_10.txt -e $experiment_config_nodf


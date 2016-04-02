#!/bin/bash

cd /home/kivan/Projects/cv-stereo/build/vo_batch_workstation/release/

experiment_config="../../../config_files/experiments/tracker_freak_7_1.txt"

./visodom -c ../../../config_files/config_kitti_11.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_12.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_13.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_14.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_15.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_16.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_17.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_18.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_19.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_20.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_21.txt -e $experiment_config


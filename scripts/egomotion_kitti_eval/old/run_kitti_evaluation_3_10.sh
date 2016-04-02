#!/bin/bash

cd /home/kivan/Projects/cv-stereo/build/vo_batch_workstation/release/

experiment_config=$1

./visodom -c ../../../config_files/config_kitti_03.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_04.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_05.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_06.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_07.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_08.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_09.txt -e $experiment_config
./visodom -c ../../../config_files/config_kitti_10.txt -e $experiment_config


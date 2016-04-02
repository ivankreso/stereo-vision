#!/bin/bash

#cd /home/kivan/Projects/cv-stereo/build/vo_debug/release/
#cd /home/kivan/Projects/cv-stereo/build/vo_batch/release/
cd /home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/

experiment_config=$1

#OMP_NUM_THREADS=22 ./visodom -c ../../../config_files/config_kitti_00.txt -e $experiment_config
#OMP_NUM_THREADS=22 ./visodom -c ../../../config_files/config_kitti_01.txt -e $experiment_config
OMP_NUM_THREADS=22 ./visodom -c ../../../config_files/config_kitti_02.txt -e $experiment_config
OMP_NUM_THREADS=22 ./visodom -c ../../../config_files/config_kitti_03.txt -e $experiment_config
OMP_NUM_THREADS=22 ./visodom -c ../../../config_files/config_kitti_04.txt -e $experiment_config
OMP_NUM_THREADS=22 ./visodom -c ../../../config_files/config_kitti_05.txt -e $experiment_config
OMP_NUM_THREADS=22 ./visodom -c ../../../config_files/config_kitti_06.txt -e $experiment_config
OMP_NUM_THREADS=22 ./visodom -c ../../../config_files/config_kitti_07.txt -e $experiment_config
OMP_NUM_THREADS=22 ./visodom -c ../../../config_files/config_kitti_08.txt -e $experiment_config
OMP_NUM_THREADS=22 ./visodom -c ../../../config_files/config_kitti_09.txt -e $experiment_config
OMP_NUM_THREADS=22 ./visodom -c ../../../config_files/config_kitti_10.txt -e $experiment_config


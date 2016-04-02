#!/bin/bash

cd /home/kivan/source/cv-stereo/build/egomotion/release/

experiment_config=$1

OMP_NUM_THREADS=10 ./egomotion -c /home/kivan/source/cv-stereo/config_files/validation_config_kitti_04.txt -e $experiment_config
OMP_NUM_THREADS=10 ./egomotion -c /home/kivan/source/cv-stereo/config_files/validation_config_kitti_05.txt -e $experiment_config
OMP_NUM_THREADS=10 ./egomotion -c /home/kivan/source/cv-stereo/config_files/validation_config_kitti_06.txt -e $experiment_config
OMP_NUM_THREADS=10 ./egomotion -c /home/kivan/source/cv-stereo/config_files/validation_config_kitti_07.txt -e $experiment_config
OMP_NUM_THREADS=10 ./egomotion -c /home/kivan/source/cv-stereo/config_files/validation_config_kitti_08.txt -e $experiment_config
OMP_NUM_THREADS=10 ./egomotion -c /home/kivan/source/cv-stereo/config_files/validation_config_kitti_09.txt -e $experiment_config
OMP_NUM_THREADS=10 ./egomotion -c /home/kivan/source/cv-stereo/config_files/validation_config_kitti_10.txt -e $experiment_config


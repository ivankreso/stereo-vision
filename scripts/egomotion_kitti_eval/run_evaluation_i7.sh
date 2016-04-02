#!/bin/bash

cd ~/source/cv-stereo/build/egomotion/release/

experiment_config=$1

export OPENBLAS_NUM_THREADS=1
./egomotion -c ../../../config_files/config_kitti_00.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_01.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_02.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_03.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_04.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_05.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_06.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_07.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_08.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_09.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_10.txt -e $experiment_config


#!/bin/bash

cd /home/kivan/source/cv-stereo/build/egomotion/release/

experiment_config=$1

#export OMP_NUM_THREADS=23
#export OMP_NUM_THREADS=10
#./egomotion -c ../../../config_files/config_kitti_00.txt -e $experiment_config
#./egomotion -c ../../../config_files/config_kitti_01.txt -e $experiment_config
#./egomotion -c ../../../config_files/config_kitti_02.txt -e $experiment_config
#./egomotion -c ../../../config_files/config_kitti_03.txt -e $experiment_config
#./egomotion -c ../../../config_files/config_kitti_04.txt -e $experiment_config
#./egomotion -c ../../../config_files/config_kitti_05.txt -e $experiment_config
#./egomotion -c ../../../config_files/config_kitti_06.txt -e $experiment_config
#./egomotion -c ../../../config_files/config_kitti_07.txt -e $experiment_config
#./egomotion -c ../../../config_files/config_kitti_08.txt -e $experiment_config
#./egomotion -c ../../../config_files/config_kitti_09.txt -e $experiment_config
#./egomotion -c ../../../config_files/config_kitti_10.txt -e $experiment_config

#export OMP_NUM_THREADS=8
##unset OMP_NUM_THREADS
#./egomotion -c ../../../config_files/config_kitti_00.txt -e $experiment_config &
#./egomotion -c ../../../config_files/config_kitti_02.txt -e $experiment_config &
#./egomotion -c ../../../config_files/config_kitti_08.txt -e $experiment_config &
#./egomotion -c ../../../config_files/config_kitti_05.txt -e $experiment_config &
#wait
#export OMP_NUM_THREADS=4
#./egomotion -c ../../../config_files/config_kitti_01.txt -e $experiment_config &
#./egomotion -c ../../../config_files/config_kitti_03.txt -e $experiment_config &
#./egomotion -c ../../../config_files/config_kitti_06.txt -e $experiment_config &
#./egomotion -c ../../../config_files/config_kitti_07.txt -e $experiment_config &
#./egomotion -c ../../../config_files/config_kitti_09.txt -e $experiment_config &
#./egomotion -c ../../../config_files/config_kitti_10.txt -e $experiment_config &
#wait
#export OMP_NUM_THREADS=23
#./egomotion -c ../../../config_files/config_kitti_04.txt -e $experiment_config &
#wait

export OMP_NUM_THREADS=10
./egomotion -c ../../../config_files/config_kitti_11.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_12.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_13.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_14.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_15.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_16.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_17.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_18.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_19.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_20.txt -e $experiment_config
./egomotion -c ../../../config_files/config_kitti_21.txt -e $experiment_config


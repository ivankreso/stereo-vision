#!/bin/bash

cd /home/kivan/Projects/cv-stereo/build/vo_sba/release/

./visodom -c ../../../config_files/config_kitti_00.txt
./visodom -c ../../../config_files/config_kitti_01.txt
./visodom -c ../../../config_files/config_kitti_02.txt
./visodom -c ../../../config_files/config_kitti_03.txt
./visodom -c ../../../config_files/config_kitti_04.txt
./visodom -c ../../../config_files/config_kitti_05.txt
./visodom -c ../../../config_files/config_kitti_06.txt
./visodom -c ../../../config_files/config_kitti_07.txt
./visodom -c ../../../config_files/config_kitti_08.txt
./visodom -c ../../../config_files/config_kitti_09.txt
./visodom -c ../../../config_files/config_kitti_10.txt


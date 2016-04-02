#!/bin/zsh

#./run_evaluation_i7.sh "/home/kivan/source/cv-stereo/config_files/experiments/kitti/ba5_Cauchy_0.15.txt"
#./run_evaluation_i7.sh "/home/kivan/source/cv-stereo/config_files/experiments/kitti/ba6_Cauchy_0.15.txt" &
#./run_evaluation_i7.sh "/home/kivan/source/cv-stereo/config_files/experiments/kitti/loss_squared.txt" &

#./run_evaluation_i7.sh "/home/kivan/source/cv-stereo/config_files/experiments/kitti/sparse_ba5_Cauchy_0.15.txt" &

export OMP_NUM_THREADS=4
#./run_evaluation_i7.sh "/home/kivan/source/cv-stereo/config_files/experiments/kitti/loss_squared_libviso_wgt_thr2.txt"
#./run_evaluation_i7.sh "/home/kivan/source/cv-stereo/config_files/experiments/kitti/loss_squared_libviso_wgt.txt"
./run_evaluation_i7.sh "/home/kivan/source/cv-stereo/config_files/experiments/kitti/orb_02.txt"

#export OMP_NUM_THREADS=4
#./run_evaluation_i7.sh "/home/kivan/source/cv-stereo/config_files/experiments/kitti/loss_squared_wgt.txt" &
#./run_evaluation_i7.sh "/home/kivan/source/cv-stereo/config_files/experiments/kitti/ba6_Cauchy_0.15_wgt.txt"

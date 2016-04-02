#!/bin/bash

cd /home/kivan/Projects/cv-stereo/build/egomotion/release/

dataset_config="../../../config_files/config_kitti_07.txt"
experiment_config="../../../config_files/experiments/kitti/ncc_validation_wgt.txt"

# the best 2 processes x 12 threads
OMP_NUM_THREADS=12 ./egomotion -c $dataset_config -e $experiment_config

#OMP_NUM_THREADS=24 ./egomotion -c $dataset_config -e $experiment_config
#OMP_NUM_THREADS=23 ./egomotion -c $dataset_config -e $experiment_config
#OMP_NUM_THREADS=22 ./egomotion -c $dataset_config -e $experiment_config
#OMP_NUM_THREADS=21 ./egomotion -c $dataset_config -e $experiment_config
#OMP_NUM_THREADS=20 ./egomotion -c $dataset_config -e $experiment_config
#OMP_NUM_THREADS=16 ./egomotion -c $dataset_config -e $experiment_config
#OMP_NUM_THREADS=14 ./egomotion -c $dataset_config -e $experiment_config
#OMP_NUM_THREADS=12 ./egomotion -c $dataset_config -e $experiment_config
#OMP_NUM_THREADS=10 ./egomotion -c $dataset_config -e $experiment_config
#OMP_NUM_THREADS=8 ./egomotion -c $dataset_config -e $experiment_config
#OMP_NUM_THREADS=4 ./egomotion -c $dataset_config -e $experiment_config
#OMP_NUM_THREADS=2 ./egomotion -c $dataset_config -e $experiment_config
#OMP_NUM_THREADS=1 ./egomotion -c $dataset_config -e $experiment_config

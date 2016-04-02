#!/bin/bash

cd /home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/

experiment_config="../../../config_files/experiments/kitti/ncc_validation_df.txt"
#experiment_config="../../../config_files/experiments/kitti/ncc_validation_df_1.txt"
#experiment_config="../../../config_files/experiments/kitti/ncc_validation_df_wgt.txt"
#experiment_config="../../../config_files/experiments/kitti/tracker_ncc_2FO-CC_2.txt"

#experiment_config="../../../config_files/experiments/kitti/ncc_validation_wgt.txt"
df_folder="/home/kivan/Dropbox/experiment_data/"

OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_00.txt -e $experiment_config -d $df_folder/00_df.yml

OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_11.txt -e $experiment_config -d $df_folder/04_df.yml
OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_12.txt -e $experiment_config -d $df_folder/04_df.yml

OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_13.txt -e $experiment_config -d $df_folder/00_df.yml
OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_14.txt -e $experiment_config -d $df_folder/00_df.yml
OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_15.txt -e $experiment_config -d $df_folder/00_df.yml

#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_04.txt -e $experiment_config -o ./results/ncc_uniform_dense_df_3.0_wgt/ -d $df_folder/without_4_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_05.txt -e $experiment_config -o ./results/ncc_uniform_dense_df_3.0_wgt/ -d $df_folder/without_5_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_06.txt -e $experiment_config -o ./results/ncc_uniform_dense_df_3.0_wgt/ -d $df_folder/without_6_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_07.txt -e $experiment_config -o ./results/ncc_uniform_dense_df_3.0_wgt/ -d $df_folder/without_7_df_3.0.yml

#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_04.txt -e $experiment_config -d $df_folder/without_4_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_05.txt -e $experiment_config -d $df_folder/without_5_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_06.txt -e $experiment_config -d $df_folder/without_6_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_07.txt -e $experiment_config -d $df_folder/without_7_df_3.0.yml

#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -o ./results/dense_5_ncc_df_2.0/ -d $df_folder/without_10_df_2.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -o ./results/dense_5_ncc_df_3.5/ -d $df_folder/without_10_df_3.5.yml

#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/noransac_ncc_dense_uniform_df_0.1/ -d $df_folder/without_8_df_0.1.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/noransac_ncc_dense_uniform_df_0.5/ -d $df_folder/without_8_df_0.5.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/noransac_ncc_dense_uniform_df_1.0/ -d $df_folder/without_8_df_1.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/noransac_ncc_dense_uniform_df_1.5/ -d $df_folder/without_8_df_1.5.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/noransac_ncc_dense_uniform_df_2.0/ -d $df_folder/without_8_df_2.0.yml

#OMP_NUM_THREADS=22 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_07.txt -e $experiment_config -o ./results/no_ransac_ncc_df_3.0/ -d $df_folder/without_07_df_3.0.yml

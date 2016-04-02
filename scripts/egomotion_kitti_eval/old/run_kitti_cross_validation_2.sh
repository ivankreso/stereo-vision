#!/bin/bash

cd /home/kivan/Projects/cv-stereo/build/vo_batch_debug/release/

experiment_config="../../../config_files/experiments/kitti/ncc_validation_df.txt"
#experiment_config="../../../config_files/experiments/kitti/ncc_validation_df_1.txt"
#experiment_config="../../../config_files/experiments/kitti/ncc_validation_df_wgt.txt"
#experiment_config="../../../config_files/experiments/kitti/tracker_ncc_2FO-CC_2.txt"
#experiment_config="../../../config_files/experiments/kitti/ncc_validation.txt"
df_folder="/home/kivan/Dropbox/experiment_data/"

OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_16.txt -e $experiment_config -d $df_folder/00_df.yml
OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_17.txt -e $experiment_config -d $df_folder/00_df.yml
OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_18.txt -e $experiment_config -d $df_folder/00_df.yml
OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_19.txt -e $experiment_config -d $df_folder/00_df.yml
OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_20.txt -e $experiment_config -d $df_folder/00_df.yml
OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_21.txt -e $experiment_config -d $df_folder/00_df.yml

#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/ncc_uniform_dense_df_3.0_wgt/ -d $df_folder/without_8_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_09.txt -e $experiment_config -o ./results/ncc_uniform_dense_df_3.0_wgt/ -d $df_folder/without_9_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -o ./results/ncc_uniform_dense_df_3.0_wgt/ -d $df_folder/without_10_df_3.0.yml

#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -d $df_folder/without_8_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_09.txt -e $experiment_config -d $df_folder/without_9_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -d $df_folder/without_10_df_3.0.yml

#OMP_NUM_THREADS=22 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/dense_5_ncc_df_3.0/ -d $df_folder/without_8_df_3.0.yml

#OMP_NUM_THREADS=22 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/noransac_ncc_best_df_3.0/ -d $df_folder/without_8_df_3.0.yml

#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/noransac_ncc_dense_uniform_df_2.5/ -d $df_folder/without_8_df_2.5.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/noransac_ncc_dense_uniform_df_3.0/ -d $df_folder/without_8_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/noransac_ncc_dense_uniform_df_3.5/ -d $df_folder/without_8_df_3.5.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/noransac_ncc_dense_uniform_df_4.0/ -d $df_folder/without_8_df_4.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/noransac_ncc_dense_uniform_df_4.5/ -d $df_folder/without_8_df_4.5.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_08.txt -e $experiment_config -o ./results/noransac_ncc_dense_uniform_df_5.0/ -d $df_folder/without_8_df_5.0.yml

#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -o ./results/dense_5_ncc_df_3.0/ -d $df_folder/without_10_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -o ./results/dense_5_ncc_df_4.0/ -d $df_folder/without_10_df_4.0.yml

#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -o ./results/bf_5_ncc_df_2.5/ -d $df_folder/without_10_df_2.5.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -o ./results/bf_5_ncc_df_3.0/ -d $df_folder/without_10_df_3.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -o ./results/bf_5_ncc_df_3.5/ -d $df_folder/without_10_df_3.5.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -o ./results/bf_5_ncc_df_4.0/ -d $df_folder/without_10_df_4.0.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -o ./results/bf_5_ncc_df_4.5/ -d $df_folder/without_10_df_4.5.yml
#OMP_NUM_THREADS=11 ./visodom -c /home/kivan/Projects/cv-stereo/config_files/config_kitti_10.txt -e $experiment_config -o ./results/bf_5_ncc_df_5.0/ -d $df_folder/without_10_df_5.0.yml

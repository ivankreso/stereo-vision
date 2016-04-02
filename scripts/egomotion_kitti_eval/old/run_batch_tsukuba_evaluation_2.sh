#!/bin/bash

cd /home/kivan/Projects/cv-stereo/build/vo_debug/release/

tsukuba_config="../../../config_files/config_tsukuba_fluorescent.txt"

./visodom -c $tsukuba_config -e ../../../config_files/experiments/tsukuba/tracker_ncc_tsukuba_1.txt
./visodom -c $tsukuba_config -e ../../../config_files/experiments/tsukuba/tracker_ncc_tsukuba_2.txt
./visodom -c $tsukuba_config -e ../../../config_files/experiments/tsukuba/tracker_ncc_tsukuba_3.txt
./visodom -c $tsukuba_config -e ../../../config_files/experiments/tsukuba/tracker_ncc_tsukuba_4.txt

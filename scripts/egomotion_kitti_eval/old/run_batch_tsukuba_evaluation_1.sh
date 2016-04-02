#!/bin/bash

cd /home/kivan/Projects/cv-stereo/build/vo_debug/release/

tsukuba_config="../../../config_files/config_tsukuba_fluorescent.txt"

./visodom -c $tsukuba_config -e ../../../config_files/experiments/tsukuba/tracker_ncc_refiner_tsukuba_3.txt
./visodom -c $tsukuba_config -e ../../../config_files/experiments/tsukuba/tracker_ncc_refiner_tsukuba_4.txt


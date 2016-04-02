#!/bin/bash

cd $2

tsukuba_config="/home/kivan/Projects/cv-stereo/config_files/config_tsukuba_fluorescent.txt"
experiment_config=$1

echo "./visodom -c $tsukuba_config -e $experiment_config"
./visodom -c $tsukuba_config -e $experiment_config

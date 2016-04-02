#!/bin/bash

cd /home/kivan/source/cv-stereo/build/learn_deformation_field/release/

#loss_scale=4.5
loss_scale=3.0
./learn_df -skip 4 -scale $loss_scale
./learn_df -skip 5 -scale $loss_scale
./learn_df -skip 6 -scale $loss_scale
./learn_df -skip 7 -scale $loss_scale
./learn_df -skip 8 -scale $loss_scale
./learn_df -skip 9 -scale $loss_scale
./learn_df -skip 10 -scale $loss_scale

#./learn_df -skip 8 -scale 0.1
#./learn_df -skip 8 -scale 0.5
#./learn_df -skip 8 -scale 1.0
#./learn_df -skip 8 -scale 1.5
#./learn_df -skip 8 -scale 2.0
#./learn_df -skip 8 -scale 2.5
#./learn_df -skip 8 -scale 3.0
#./learn_df -skip 8 -scale 3.5
#./learn_df -skip 8 -scale 4.0
#./learn_df -skip 8 -scale 4.5
#./learn_df -skip 8 -scale 5.0


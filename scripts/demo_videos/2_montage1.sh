#!/bin/bash

input_left="/home/kivan/Projects/demo/vo_demo/left_cam/"
input_right="/home/kivan/Projects/demo/vo_demo/right_cam/"
out_folder="montage_1/"

for i in {0..928}
do
   #echo i
   printf -v filename_left "img_left_%06d.jpg" $i
   printf -v filename_right "img_right_%06d.jpg" $i
   path_left=$input_left$filename_left
   path_right=$input_right$filename_right
   echo $path_left
   printf -v out_img $out_folder"img_%06d.jpg" $i
   montage -mode concatenate -tile 1x $path_left $path_right $out_img
done

#!/bin/bash

input_plot="demo_plot_crop/"
input_cam="montage_1/"
out_folder="montage_2/"

for i in {0..928}
do
   printf -v filename_plot "img_plot_%06d.png" $i
   printf -v filename_cam "img_%06d.jpg" $i
   path_plot=$input_plot$filename_plot
   path_cam=$input_cam$filename_cam
   echo $path_plot
   printf -v out_img $out_folder"img_%06d.jpg" $i
   montage -mode concatenate -tile x1 $path_plot $path_cam $out_img
done

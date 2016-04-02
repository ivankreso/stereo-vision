#!/bin/bash

input_folder="demo_plot/"
out_folder="demo_plot_crop/"

for i in {0..928}
do
   #echo i
   printf -v filename "img_plot_%06d.png" $i
   path_img=$input_folder$filename
   echo $path_img
   printf -v out_img $out_folder"img_plot_%06d.png" $i
   convert -crop 740x927+130+0 +repage $path_img $out_img
done

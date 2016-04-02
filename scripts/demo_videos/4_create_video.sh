#!/bin/bash

ffmpeg -f image2 -r 16 -i montage_video/img_%06d.jpg -c:v libx264 -crf 18 -r 16 demo_vo.avi

ffmpeg -f image2 -r 16 -i montage_video/img_%06d.jpg -c:v mpeg4 -b 8000k -r 16 demo_vo_mpeg4.avi

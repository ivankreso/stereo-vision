import numpy as np
import cv2

input_path = "/home/kivan/Projects/cv-stereo/build/vo_sba/release/vo.txt"
output_file = "motion_init.txt"


path_Rt = np.loadtxt(input_path)

nframes = path_Rt.shape[0]

for i in range(nframes):
    cv2.Rodrigues()

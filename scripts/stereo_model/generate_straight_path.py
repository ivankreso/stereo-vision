import numpy as np

def write_mat(Rt, fp):
   for i in range(Rt.shape[0]-1):
      for j in range(Rt.shape[1]):
         fp.write(str(Rt[i,j]) + " ")
   fp.write("\n")

fp = open("path.txt", "w")
length = 175.0 # meters
nframes = 632-446 + 1

delta_z = length / nframes

Rt = np.eye(4,4)

for i in range(nframes):
   write_mat(Rt, fp)
   Rt[2,3] += delta_z



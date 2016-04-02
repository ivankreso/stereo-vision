import numpy as np

def write_mat(Rt, fp):
    for i in range(Rt.shape[0]-1):
        for j in range(Rt.shape[1]):
            fp.write(str(Rt[i,j]) + " ")
    fp.write("\n")

def write_vec(vec, fp):
    for i in range(vec.shape[0]):
        fp.write(str(vec[i]) + " ")
    fp.write("\n")

fp = open("path_noise.txt", "w")
#fp = open("path.txt", "w")
#length = 10.0       # meters
length = 7.0       # meters
#nframes = 40
nframes = 20

delta_z = length / nframes

#Rt = np.eye(4,4)

# using Rodrigues rotation
vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

for i in range(nframes):
    #write_mat(Rt, fp)
    #Rt[2,3] += delta_z
    write_vec(vec, fp)
    vec[5] -= delta_z

#import sys
from numpy import *


def quat_to_transmat(Q):
   w = Q[0,0]
   x = Q[0,1]
   y = Q[0,2]
   z = Q[0,3]
   tx = Q[0,4]
   ty = Q[0,5]
   tz = Q[0,6]
   # left handed or right? post/pre-multiply?
   T = matrix([[1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,	 2*x*z + 2*y*w,	      tx],
               [2*x*y + 2*z*w,	    1 - 2*x*x - 2*z*z,	 2*y*z - 2*x*w,	      ty],
               [2*x*z - 2*y*w,	    2*y*z + 2*x*w,	 1 - 2*x*x - 2*y*y,   tz]])
   #T[:,0:3] = T[:,0:3].T
   #T[:,3] = -T[:,3]
   return T


set_printoptions(precision=2)

#savetxt('K.txt', K, fmt='%.5f')
# only for array type txt file
#K = matrix(fromfile("K.txt", sep='\n').reshape(3,3))

# load camera matrix C
#C = eye(3)
#savetxt('C.txt', C, fmt='%.2f')
C = matrix(loadtxt('C.txt'))
print('C:\n', C, '\n')

# load 3D points
#pts3d = matrix([[0,0,6,1], [1,1,6,1], [-4,4,6,1], [2,-3,6,1], [3,3,7,1], [5,4,7,1], [-1,-2,7,1], [-2,5,7,1], [1,1,8,1], [2,2,8,1], [3,-3,8,1], [6,7,8,1]]).transpose()
#pts3d = matrix([[0,0,6,1], [5,5,6,1], [-4,10,6,1], [20,-3,6,1], [9,3,7,1], [25,11,7,1], [-10,-22,7,1],
#	        [-4,15,7,1], [10,11,8,1], [22,12,8,1], [23,-13,8,1], [16,7,8,1]]).transpose()
#savetxt('pts3d.txt', pts3d, fmt='%.2f')
pts3d = matrix(loadtxt('pts3d.txt'))
print('3D points:\n', pts3d, '\n')

# load camera movement transformations
#rt = matrix([[1,0,0,0,0,0,0], [1,0,0,0,0,0,1], [1,0,0,0,1,0,2], [1,0,0,0,2,0,3], [1,0,0,0,3,0,4]])
#savetxt('rt.txt', rt, fmt='%.2f')
rt = matrix(loadtxt('rt_rot.txt'))
print('camera rotation-translation params:\n', rt, '\n')

[r, c] = rt.shape
[d, npts] = pts3d.shape
nframes = r
projs = zeros([nframes, 3, npts])
for i in range(nframes):
   T = quat_to_transmat(rt[i,:])
   print("using transformation:\n", T, '\n')
   pts2d = C * T * pts3d
   pts2d = pts2d / pts2d[2,:]
   projs[i,:,:] = pts2d
   #append(projs, pts2d)
   print('Frame ', i, ':\n', pts2d, '\n')

print("All projections: \n")
print(projs)

fp = open("pts_data.txt", "w")
SPLIT_PTS = 2
for i in range(npts-SPLIT_PTS):
   fp.write(str(pts3d[0,i]) + " " + str(pts3d[1,i]) + " " + str(pts3d[2,i]) + " " + str(nframes))
   for f in range(nframes):
      fp.write(" " + str(f) + " " + str(projs[f,0,i]) + " " + str(projs[f,1,i]))
   fp.write("\n")

split_frames = 3
for i in range(npts-SPLIT_PTS, npts):
   fp.write(str(pts3d[0,i]) + " " + str(pts3d[1,i]) + " " + str(pts3d[2,i]) + " " + str(nframes-split_frames))
   for f in range(split_frames, nframes):
      fp.write(" " + str(f) + " " + str(projs[f,0,i]) + " " + str(projs[f,1,i]))
   fp.write("\n")

fp.close()


# now run: ./eucsbademo rt_pts.txt pts_data.txt C.txt 

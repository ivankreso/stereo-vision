#!/bin/python

import numpy as np

# generate 3D points around points in pts_center using gaussian distribution
def generate_points(pts_center, pts_sigma, pts_fixed, pts_num, axis_range):
    # allocate mem for points
    points = np.zeros([4, pts_num.sum()])
    istart = 0
    for i in range(pts_center.shape[1]):
        #print(pts_center[:,i])
        for j in range(pts_num[i]):
            for k in range(3):
                #print(i,j,k)
                if pts_fixed[k,i] == 0:
                    points[k,istart+j] = np.random.normal(pts_center[k,i], pts_sigma[k,i])
                else:
                    points[k,istart+j] = pts_center[k,i]
                # force axis range if outside of domain
                if points[k,istart+j] < axis_range[k,0]:
                    points[k,istart+j] = axis_range[k,0]
                elif points[k,istart+j] > axis_range[k,1]:
                    points[k,istart+j] = axis_range[k,1]

            points[3,istart+j] = 1.0
        istart += pts_num[i]
    #print(points)
    return points

def main():
    # axis domains
    range_x = [-30, 30]
    range_y = [-20, 8] # -20, 3
    range_z = [5, 150]
    # best
    #range_x = [-30, 30]
    #range_y = [-20, 8] # -20, 3
    #range_z = [10, 150]
    axis_range = np.array([range_x, range_y, range_z])
    # point centroids
    # close
    #pts_center = np.array([[-5, -2, 20, 1], [5, -2, 20, 1], [0, -2, 20, 1]]).T
    # far
    #pts_center = np.array([[-10, -5, 60, 1], [10, -5, 60, 1], [0, -5, 80, 1]]).T
    #pts_center = np.array([[-15, -2, 80, 1], [15, -2, 80, 1], [0, -2, 80, 1]]).T
    # best
    #pts_center = np.array([[-10, -2, 60, 1], [10, -2, 60, 1], [0, -2, 90, 1]]).T
    pts_center = np.array([[-10, -2, 30, 1], [10, -2, 30, 1], [0, -2, 60, 1]]).T
    # coords with fixed values are marked with 1
    #pts_fixed = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]]).transpose()
    pts_fixed = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).T
    # max and min range of points coords
    #pts_range = np.array([range_x, range_y, range_z])
    # sigma value for gauss distribution
    # close
    pts_sigma = np.array([[3, 4, 10], [6, 4, 10], [6, 4, 20]]).T
    # far
    #pts_sigma = np.array([[2, 4, 20], [2, 4, 20], [4, 4, 20]]).T
    #pts_sigma = np.array([[10, 4, 20], [10, 4, 20], [10, 4, 20]]).T
    # best
    #pts_sigma = np.array([[6, 6, 20], [6, 6, 20], [10, 6, 20]]).T

    #pts_num = np.array([25, 25, 40])
    pts_num = np.array([50, 50, 100])
    pts3d = generate_points(pts_center, pts_sigma, pts_fixed, pts_num, axis_range)
    np.savetxt("pts3d.txt", pts3d)

if __name__ == "__main__":
    main()

import sys
from numpy import *

set_printoptions(precision=2)

def add_noise_gaussian(M, sigma):
   N = random.normal(0, sigma, M.shape)
   M += N

def add_noise_uniform(M, rng):
   N = (random.rand(M.shape) * rng) - (rng/2.0)
   M += N

def main():
   rt = matrix(loadtxt("rt.txt"))
   print("original params:\n", rt, "\n")
   add_noise_gaussian(rt, 0.1)
   #add_noise_uniform(rt, 0.4)
   print("after adding gaussian noise\n", rt, "\n")
   savetxt(sys.argv[1][:-4] + "_noised.txt", rt, fmt="%.2f")


if __name__ == '__main__':
   main()

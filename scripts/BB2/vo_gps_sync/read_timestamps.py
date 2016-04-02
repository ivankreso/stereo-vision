#!/usr/bin/python
import os
import math
import re
import struct


src_folder = '/home/kreso/projects/master_thesis/datasets/bumblebee/20121031/'
#src_folder = "/home/kreso/Downloads/pics/"

def readstamp(f):
   pgmoffset=17        # for stereo data
   #pgmoffset=15         # for mono data
   bs=f.read(pgmoffset+4)
   #x=struct.unpack(">I",bs[pgmoffset:pgmoffset+4])[0]
   x=struct.unpack("<I",bs[pgmoffset:pgmoffset+4])[0]    # reverse byte order - wrong
   t = (x>>0)  & 0xffffffff
   t = ((t >> 16) & 0xffff) | ((t << 16) & 0xffff0000)
   return t

def getTime(gps_pt):
   return gps_pt[10]*60 + gps_pt[11]

t_prev=-1
deltas=[]
for name in sorted(os.listdir(src_folder)):
   m=re.match(r'fc2.*pgm', name)
   if m:
      t = readstamp(open(src_folder+name, mode='rb'))
      delta=t-t_prev if t_prev>=0 else 0
      if delta<0:
         delta+=65536
      #print('{} {:08x} {:016b} {:016b}'.format(name, t, t >> 16, t & 0xffff))
      print('{} {:08x} {:07b} {:013b} {:012b}'.format(name, t, t >> 25, (t >> 12) & 0x1fff, t & 0xfff))
      t_prev=t
      deltas.append(delta)


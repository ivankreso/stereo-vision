#!/usr/bin/python

import os
import re
import struct

def readstamp(f):
  pgmoffset=17
  bs=f.read(pgmoffset+4)
  x=struct.unpack("<I",bs[pgmoffset:pgmoffset+4])[0]
  w = (x>>20) & 0x0fff
  n = (x>>16) & 0x000f
  t = (x>>0)  & 0xffff
  return w,n,t

t_prev=-1
deltas=[]
for name in sorted(os.listdir('.')):
  m=re.match(r'fc2.*pgm', name)
  if m:
    w,n,t=readstamp(open(name, mode='rb'))
    delta=t-t_prev if t_prev>=0 else 0
    if delta<0:
      delta+=65536
    print('{} {:01x} {:04x} {}'.format(name, n,t, delta))
    t_prev=t
    deltas.append(delta)

print(sum(deltas)/len(deltas))
for x in sorted(set(deltas)):
  print (x, deltas.count(x))

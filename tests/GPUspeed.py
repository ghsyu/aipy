#!/usr/bin/env python

import numpy as n
import aipy as a
import aipy.deconv as d1
import aipy.deconvGPU as d2

SIZEX = 1024
SIZEY = 2048
aim = n.zeros((SIZEX,SIZEY), dtype=n.complex64)
aim[10,10] = 10.
aim[20:25,20:25] = 1j
aim[30:40,30:40] = .1+.1j
dbm = a.img.gaussian_beam(2, shape=aim.shape).astype(n.complex64)
dbm[0,0] = 2
dbm /= dbm.sum()
dbm = n.ascontiguousarray(dbm)
#dbm = n.random.normal(size=dbm.shape)
dim = n.fft.ifft2(n.fft.fft2(aim) * n.fft.fft2(dbm)).astype(n.complex64)
#dim = n.random.normal(size=dim.shape)
#print 'COMPLEX TEST:'
#print("\nCPU version\n")
#i=0
#while i<10:
#    cim0, info = d1.clean(dim, dbm, tol=1e-5, stop_if_div=True, maxiter=1000)
#    i+=1
#print info
print("\nGPU version\n")
i=0
while i<10:
    cim, info = d2.clean(dim, dbm, tol=1e-5, stop_if_div=True, maxiter=1000)
    i+=1
print info

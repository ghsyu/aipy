#!/usr/bin/env python

import numpy as n
import aipy as a
import aipy.deconv as d1
import aipy.deconvGPU as d2
import time

SIZEX = 1024
SIZEY = 2048

class Test(object):
    def __init__(self, iterations, devices = [0,1,2,3]):
        self.aim = n.zeros((SIZEX,SIZEY), dtype=n.complex64)
        self.aim[10,10] = 10.
        self.aim[20:25,20:25] = 1j
        self.aim[30:40,30:40] = .1+.1j
        self.dbm = a.img.gaussian_beam(2, shape=self.aim.shape).astype(n.complex64)
        self.dbm[0,0] = 2
        self.dbm /= self.dbm.sum()
        self.dbm = n.ascontiguousarray(self.dbm)
        #dbm = n.random.normal(size=dbm.shape)
        self.dim = n.fft.ifft2(n.fft.fft2(self.aim) * n.fft.fft2(self.dbm)).astype(n.complex64)
        self.iterations = iterations
        self.devices = devices
        self.ims = n.array([self.dim]*len(self.devices))
    def run(self):
        for i in xrange(self.iterations):
            self.cim, self.info = d2.clean(self.ims, self.dbm, tol=1e-5, stop_if_div=True, maxiter=1000, devices=self.devices)

if __name__ == '__main__':
    
    A = Test(4, [0,1,2,3])
    start = time.time()
    A.run()
    print time.time()-start
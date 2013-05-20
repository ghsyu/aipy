#!/usr/bin/env python

import numpy as n
import aipy as a
import aipy.deconv as d1
import aipy.deconvGPU as d2
import threading
import cProfile

SIZEX = 1024
SIZEY = 2048

class Test(object):
    def __init__(self):
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
    
    def run_CPU(self, n):
        for i in xrange(n):
            self.cim, self.info = d1.clean(self.dim, self.dbm, tol=1e-5, stop_if_div=True, maxiter=1000)
        
    def run(self, n):
        for i in xrange(n):
            self.cim, self.info = d2.clean(self.dim, self.dbm, tol=1e-5, stop_if_div=True, maxiter=1000)
    
if __name__ == '__main__':
    def profile():
        A = Test()
        A.run(10)

    t0 = threading.Thread(target = profile())
    t1 = threading.Thread(target = profile())
    t2 = threading.Thread(target = profile())
    t0.start()
    t1.start()
    t2.start()
    t0.join()
    t1.join()
    t2.join()

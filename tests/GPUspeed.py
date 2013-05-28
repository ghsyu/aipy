#!/usr/bin/env python

import numpy as n
import aipy as a
import aipy.deconv as d1
import aipy.deconvGPU as d2
import threading
import cProfile

SIZEX = 1024
SIZEY = 2048

class Test(threading.Thread):
    def __init__(self, iterations):
        super(Test, self).__init__()
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
    def run(self):
        for i in xrange(self.iterations):
            self.cim, self.info = d2.clean(self.dim, self.dbm, tol=1e-5, stop_if_div=True, maxiter=1000)

class FastTest(Test):
    def run(self):
        for i in xrange(self.iterations):
            self.cim, self.info = d2.clean(self.dim, self.dbm, tol=1e-5, stop_if_dev=True, maxiter = 10)

if __name__ == '__main__':
    
    t0 = Test(2)
    t1 = Test(2)
    t2 = Test(2)
    
    t0.start()
    t1.start()
    t2.start()
    t0.join()
    t1.join()
    t2.join()
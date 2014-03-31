"""
A module implementing various techniques for deconvolving an image by a
kernel.  Currently implemented are Clean, Least-Squares, Maximum Entropy,
and Annealing.  Standard parameters to these functions are:
im = image to be deconvolved.
ker = kernel to deconvolve by (must be same size as im).
mdl = a priori model of what the deconvolved image should look like.
maxiter = maximum number of iterations performed before terminating.
tol = termination criterion, lower being more optimized.
verbose =  print info on how things are progressing.
lower = lower bound of pixel values in deconvolved image
upper = upper bound of pixel values in deconvolved image
"""

import numpy as n, sys, _deconvGPU
import pyfftw as pf

# Find smallest representable # > 0 for setting clip level
lo_clip_lev = n.finfo(n.float).tiny 

def clean(ims, ker, mdl=None, area=None, devices=[0, 1, 2, 3], gain=.1, maxiter=10000, tol=1e-3, 
        stop_if_div=True, verbose=False, pos_def=False):
    """This standard Hoegbom clean deconvolution algorithm operates on the 
    assumption that the image is composed of point sources.  This makes it a 
    poor choice for images with distributed flux.  In each iteration, a point 
    is added to the model at the location of the maximum residual, with a 
    fraction (specified by 'gain') of the magnitude.  The convolution of that 
    point is removed from the residual, and the process repeats.  Termination 
    happens after 'maxiter' iterations, or when the clean loops starts 
    increasing the magnitude of the residual.  This implementation can handle 
    1 and 2 dimensional data that is real valued or complex.
    gain: The fraction of a residual used in each iteration.  If this is too
    low, clean takes unnecessarily long.  If it is too high, clean does
    a poor job of deconvolving."""
    if len(ims) != len(devices):
        raise ValueError('number of images must equal number of devices')
    if mdl == None:
        mdl = [None]*len(devices)
    mdl_l = []
    res_l = []
    info_l = []
    if len(ker) != len(devices):
        if len(ker) == len(ims[0]):
            ker_l = [n.ascontiguousarray(ker).astype(n.complex64)]*len(devices)
        else:
            raise ValueError('size of kernel must equal size of image')
    else:
        ker_l = []
        for k in ker:
            ker_l.append(n.ascontiguousarray(k).astype(n.complex64))
    ims = iter(ims)
    kers = iter(ker_l)
    for m in mdl:
        k = kers.next()
        im = ims.next().astype(n.complex64)
        if m is None:
            m = n.zeros(im.shape, dtype=im.dtype)
            res = im.copy()
        else:
            m = m.copy().astype(n.complex64)
            if len(m.shape) == 1:
                res = im - pf.interfaces.numpy_fft.ifft(pf.interfaces.numpy_fft.fft(m) * \
                                      pf.interfaces.numpy_fft.fft(k)).astype(im.dtype)
            elif len(m.shape) == 2:
                res = im - pf.interfaces.numpy_fft.ifft2(pf.interfaces.numpy_fft.fft2(m) * \
                                       pf.interfaces.numpy_fft.fft2(k)).astype(im.dtype)
            else: raise ValueError('Number of dimensions != 1 or 2')
        mdl_l.append(n.ascontiguousarray(m))
        res_l.append(n.ascontiguousarray(res))
    if area is None:
        area = n.ones(im.shape, dtype=n.int32)
    else:
        area = area.astype(n.int32)
    _iter = _deconvGPU.clean(res_l, ker_l, mdl_l, area,
            gain=gain, maxiter=maxiter, tol=tol,
            stop_if_div=int(stop_if_div), verbose=int(verbose),
            pos_def=int(pos_def), devices=devices)
    score = n.sqrt(n.average(n.abs(res)**2))
    for i in _iter:
        info = {'success':i > 0 and i < maxiter, 'tol':tol}
        if i < 0: info.update({'term':'divergence', 'iter':-i})
        elif i < maxiter: info.update({'term':'tol', 'iter':i})
        else: info.update({'term':'maxiter', 'iter':i})
        info.update({'res':res, 'score':score})
        if verbose:
            print 'Term Condition:', info['term']
            print 'Iterations:', info['iter']
            print 'Score:', info['score']
        info_l.append(info)
    return mdl_l, info_l

def recenter(a, c):
    """Slide the (0,0) point of matrix a to a new location tuple c."""
    s = a.shape
    c = (c[0] % s[0], c[1] % s[1])
    a1 = n.concatenate([a[c[0]:], a[:c[0]]], axis=0)
    a2 = n.concatenate([a1[:,c[1]:], a1[:,:c[1]]], axis=1)
    return a1

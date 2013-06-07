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
    mdl_l = []
    res_l = []
    ims = iter(ims)
    for device in devices:
        im = ims.next()
        if mdl is None:
            mdl = n.zeros(im.shape, dtype=im.dtype)
            res = im.copy()
        else:
            mdl = mdl.copy()
            if len(mdl.shape) == 1:
                res = im - n.fft.ifft(n.fft.fft(mdl) * \
                                      n.fft.fft(ker)).astype(im.dtype)
            elif len(mdl.shape) == 2:
                res = im - n.fft.ifft2(n.fft.fft2(mdl) * \
                                       n.fft.fft2(ker)).astype(im.dtype)
            else: raise ValueError('Number of dimensions != 1 or 2')
        mdl_l.append(mdl)
        res_l.append(res)
    if area is None:
        area = n.ones(im.shape, dtype=n.int32)
    else:
        area = area.astype(n.int32)
    _iter = _deconvGPU.clean(res_l, ker, mdl_l, area,
            gain=gain, maxiter=maxiter, tol=tol, 
            stop_if_div=int(stop_if_div), verbose=int(verbose),
            pos_def=int(pos_def), devices=devices)
    score = n.sqrt(n.average(n.abs(res)**2))
    info = {'success':_iter > 0 and _iter < maxiter, 'tol':tol}
    if _iter < 0: info.update({'term':'divergence', 'iter':-_iter})
    elif _iter < maxiter: info.update({'term':'tol', 'iter':_iter})
    else: info.update({'term':'maxiter', 'iter':_iter})
    info.update({'res':res, 'score':score})
    if verbose:
        print 'Term Condition:', info['term']
        print 'Iterations:', info['iter']
        print 'Score:', info['score']
        
    return mdl_l, info

def recenter(a, c):
    """Slide the (0,0) point of matrix a to a new location tuple c."""
    s = a.shape
    c = (c[0] % s[0], c[1] % s[1])
    a1 = n.concatenate([a[c[0]:], a[:c[0]]], axis=0)
    a2 = n.concatenate([a1[:,c[1]:], a1[:,:c[1]]], axis=1)
    return a2


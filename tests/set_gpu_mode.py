#!/usr/bin/env python
import pynvml as nv


def set_mode(mode):
    nv.nvmlInit()
    compute_modes = []
    #Set all devices available to Compute exclusive
    for i in range(nv.nvmlDeviceGetCount()):
        handle = nv.nvmlDeviceGetHandleByIndex(i)
        compute_modes.append(nv.nvmlDeviceGetComputeMode(handle))
        nv.nvmlDeviceSetComputeMode(handle, mode)
    nv.nvmlShutdown()
    
if __name__ == '__main__':
    import sys
    set_mode(int(sys.argv[-1]))
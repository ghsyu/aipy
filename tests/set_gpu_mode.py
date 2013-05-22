#!/usr/bin/env python
import pynvml as nv


def set_mode(mode):
    pynvml.nvmlInit()
    compute_modes = []
    #Set all devices available to Compute exclusive
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        compute_modes.append(pynvml.nvmlDeviceGetComputeMode(handle))
        pynvml.nvmlDeviceSetComputeMode(handle, mode)
    pynvml.nvmlShutdown()
    
if __name__ == '__main__':
    import sys
    set_mode(int(sys.argv[-1]))
#ifndef _CLEAN_H_
#define _CLEAN_H_


#define BLOCKSIZEX 16
#define BLOCKSIZEY 16

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

float *clean_2d_c_GPU(float *, float *, int *, double, int, \
					  float, float, int, int, int, int, int, int, int, \
					  float *, float *, float *, int *, int *, float *, float *, int*, \
					  float *, float *, int*, float *, float *, int*);

int gpu_set_up(float **, float **, int **, \
               float **, float **, int **, \
               float **, float **, int **, \
               float *, float *, int *, int, int, int, int, int);
	

int gpu_free(float *, float *, int *, \
             float *, float *, int *, \
             float *, float *, int *);


int copy_res(float *, float *, int );


#endif /* _CLEAN_H_ */

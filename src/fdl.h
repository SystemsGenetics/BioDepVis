#ifndef FDL_H
#define FDL_H

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include "vector.h"



#define CUDA_SAFE_CALL(x)                              \
{                                                      \
	cudaError_t result = x;                            \
	if ( result != cudaSuccess )                       \
	{                                                  \
		std::cerr                                      \
			<< "CUDA Error: " #x " failed with error " \
			<< cudaGetErrorString(result) << '\n';     \
		exit(1);                                       \
	}                                                  \
}



void fdl_2d_cpu(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix);
void fdl_3d_cpu(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix);
void fdl_2d_gpu(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix);
void fdl_3d_gpu(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix);



#endif

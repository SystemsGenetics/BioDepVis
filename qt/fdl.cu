#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "fdl.h"

void checkError(cudaError_t err)
{
	if ( err != cudaSuccess ) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
		exit(1);
	}
}

void * gpu_malloc(int size)
{
	void *ptr = nullptr;

	cudaMalloc(&ptr, size);
	checkError(cudaGetLastError());

	return ptr;
}

void gpu_free(void *ptr)
{
	cudaFree(ptr);
	checkError(cudaGetLastError());
}

void gpu_read(void *dst, void *src, int size)
{
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
	checkError(cudaGetLastError());
}

void gpu_write(void *dst, void *src, int size)
{
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	checkError(cudaGetLastError());
}

void fdl_2d_gpu(int n, vec3_t *coords, vec3_t *coords_d, int *edge_matrix)
{
}

void fdl_3d_gpu(int n, vec3_t *coords, vec3_t *coords_d, int *edge_matrix)
{
}

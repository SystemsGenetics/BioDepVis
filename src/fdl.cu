#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "fdl.h"



const int BLOCK_SIZE = 256;
const int MAX_DISPLACEMENT_SQR = 10.0f;



#define ELEM(data, cols, i, j) (data)[(i) * (cols) + (j)]



void checkError(cudaError_t err)
{
	if ( err != cudaSuccess )
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(err));
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



void gpu_sync()
{
	cudaDeviceSynchronize();
	checkError(cudaGetLastError());
}



__global__ void fdl_kernel_2d(int n, vec3_t *positions, vec3_t *positions_d, const bool *edge_matrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i < n )
	{
		const float K_r = 0.2f;
		const float K_s = 1.0f;
		const float L = 2.2f;

		for ( int j = 0; j < n; j++ )
		{
			if ( i == j )
			{
				continue;
			}

			float dx = positions[j].x - positions[i].x;
			float dy = positions[j].y - positions[i].y;
			float dist = sqrtf(dx * dx + dy * dy);

			if ( dist != 0 )
			{
				float force = ELEM(edge_matrix, n, i, j)
					? K_s * (L - dist) / dist
					: K_r / (dist * dist * dist);

				positions_d[i].x -= force * dx;
				positions_d[i].y -= force * dy;

				positions_d[j].x += force * dx;
				positions_d[j].y += force * dy;
			}
		}
		__syncthreads();

		float dx = positions_d[i].x;
		float dy = positions_d[i].y;
		float disp_sqr = dx * dx + dy * dy;

		if ( disp_sqr > MAX_DISPLACEMENT_SQR )
		{
			dx *= sqrtf(MAX_DISPLACEMENT_SQR / disp_sqr);
			dy *= sqrtf(MAX_DISPLACEMENT_SQR / disp_sqr);
		}

		positions[i].x += dx;
		positions[i].y += dy;
		positions_d[i].x *= 0.1f;
		positions_d[i].y *= 0.1f;
	}
}



__global__ void fdl_kernel_3d(int n, vec3_t *positions, vec3_t *positions_d, const bool *edge_matrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i < n )
	{
		const float K_r = 0.2f;
		const float K_s = 1.0f;
		const float L = 2.2f;

		for ( int j = 0; j < n; j++ )
		{
			if ( i == j )
			{
				continue;
			}

			float dx = positions[j].x - positions[i].x;
			float dy = positions[j].y - positions[i].y;
			float dz = positions[j].z - positions[i].z;
			float dist = sqrtf(dx * dx + dy * dy + dz * dz);

			if ( dist != 0 )
			{
				float force = ELEM(edge_matrix, n, i, j)
					? K_s * (L - dist) / dist
					: K_r / (dist * dist * dist);

				positions_d[i].x -= force * dx;
				positions_d[i].y -= force * dy;
				positions_d[i].z -= force * dz;

				positions_d[j].x += force * dx;
				positions_d[j].y += force * dy;
				positions_d[j].z += force * dz;
			}
		}
		__syncthreads();

		float dx = positions_d[i].x;
		float dy = positions_d[i].y;
		float dz = positions_d[i].z;
		float disp_sqr = dx * dx + dy * dy + dz * dz;

		if ( disp_sqr > MAX_DISPLACEMENT_SQR )
		{
			dx *= sqrtf(MAX_DISPLACEMENT_SQR / disp_sqr);
			dy *= sqrtf(MAX_DISPLACEMENT_SQR / disp_sqr);
			dz *= sqrtf(MAX_DISPLACEMENT_SQR / disp_sqr);
		}

		positions[i].x += dx;
		positions[i].y += dy;
		positions[i].z += dz;
		positions_d[i].x *= 0.1f;
		positions_d[i].y *= 0.1f;
		positions_d[i].z *= 0.1f;
	}
}



void fdl_2d_gpu(int n, vec3_t *positions, vec3_t *positions_d, const bool *edge_matrix)
{
	fdl_kernel_2d<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
		n,
		positions,
		positions_d,
		edge_matrix);
}



void fdl_3d_gpu(int n, vec3_t *positions, vec3_t *positions_d, const bool *edge_matrix)
{
	fdl_kernel_3d<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
		n,
		positions,
		positions_d,
		edge_matrix);
}

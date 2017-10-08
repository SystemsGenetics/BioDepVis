#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "fdl.h"

const int BLOCK_SIZE = 256;
const int MAX_DISPLACEMENT_SQR = 10.0f;

#define ELEM(data, cols, i, j) (data)[(i) * (cols) + (j)]

void checkError(cudaError_t err)
{
	if ( err != cudaSuccess ) {
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

__global__ void fdl_kernel_2d(int n, vec3_t *coords, vec3_t *coords_d, const int *edge_matrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i < n ) {
		const float K_r = 25.0f;
		const float K_s = 15.0f;
		const float L = 1.2f;
		const float dt = 0.004f;

		for ( int j = 0; j < n; j++ ) {
			if ( i == j ) {
				continue;
			}

			float dx = coords[j].x - coords[i].x;
			float dy = coords[j].y - coords[i].y;
			float dist = sqrtf(dx * dx + dy * dy);

			if ( dist != 0 ) {
				float force = (ELEM(edge_matrix, n, i, j) != 0)
					? K_s * (L - dist) / dist
					: K_r / (dist * dist * dist);

				coords_d[i].x -= force * dx;
				coords_d[i].y -= force * dy;

				coords_d[j].x += force * dx;
				coords_d[j].y += force * dy;
			}
		}

		float dx = coords_d[i].x * dt;
		float dy = coords_d[i].y * dt;
		float disp_sqr = dx * dx + dy * dy;

		if ( disp_sqr > MAX_DISPLACEMENT_SQR ) {
			dx *= sqrtf(MAX_DISPLACEMENT_SQR / disp_sqr);
			dy *= sqrtf(MAX_DISPLACEMENT_SQR / disp_sqr);
		}

		coords[i].x += dx;
		coords[i].y += dy;
		coords_d[i].x *= 0.09f;
		coords_d[i].y *= 0.09f;
	}
}

__global__ void fdl_kernel_3d(int n, vec3_t *coords, vec3_t *coords_d, const int *edge_matrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i < n ) {
		const float K_r = 25.0f;
		const float K_s = 15.0f;
		const float L = 1.2f;
		const float dt = 0.004f;

		for ( int j = 0; j < n; j++ ) {
			if ( i == j ) {
				continue;
			}

			float dx = coords[j].x - coords[i].x;
			float dy = coords[j].y - coords[i].y;
			float dz = coords[j].z - coords[i].z;
			float dist = sqrtf(dx * dx + dy * dy + dz * dz);

			if ( dist != 0 ) {
				float force = (ELEM(edge_matrix, n, i, j) != 0)
					? K_s * (L - dist) / dist
					: K_r / (dist * dist * dist);

				coords_d[i].x -= force * dx;
				coords_d[i].y -= force * dy;
				coords_d[i].z -= force * dz;

				coords_d[j].x += force * dx;
				coords_d[j].y += force * dy;
				coords_d[j].z += force * dz;
			}
		}

		float dx = coords_d[i].x * dt;
		float dy = coords_d[i].y * dt;
		float dz = coords_d[i].z * dt;
		float disp_sqr = dx * dx + dy * dy + dz * dz;

		if ( disp_sqr > MAX_DISPLACEMENT_SQR ) {
			dx *= sqrtf(MAX_DISPLACEMENT_SQR / disp_sqr);
			dy *= sqrtf(MAX_DISPLACEMENT_SQR / disp_sqr);
			dz *= sqrtf(MAX_DISPLACEMENT_SQR / disp_sqr);
		}

		coords[i].x += dx;
		coords[i].y += dy;
		coords[i].z += dz;
		coords_d[i].x *= 0.09f;
		coords_d[i].y *= 0.09f;
		coords_d[i].z *= 0.09f;
	}
}

void fdl_2d_gpu(int n, vec3_t *coords, vec3_t *coords_d, const int *edge_matrix)
{
	fdl_kernel_2d<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(n, coords, coords_d, edge_matrix);
}

void fdl_3d_gpu(int n, vec3_t *coords, vec3_t *coords_d, const int *edge_matrix)
{
	fdl_kernel_3d<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(n, coords, coords_d, edge_matrix);
}

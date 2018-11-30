#include "fdl.h"
#include <cstdint>



const float MAX_VELOCITY_MAGNITUDE_SQR = 10.0f;



#define ELEM(data, cols, i, j) (data)[(int64_t)(i) * (cols) + (j)]



__global__
void fdl_kernel_2d(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= n )
	{
		return;
	}

	// define force constants
	const float K_r = 0.2f;
	const float K_s = 1.0f;
	const float L = 2.2f;

	// get node position
	float p_x = positions[i].x;
	float p_y = positions[i].y;

	// compute node velocity
	float v_x = velocities[i].x;
	float v_y = velocities[i].y;

	for ( int j = 0; j < n; j++ )
	{
		float dx = positions[j].x - p_x;
		float dy = positions[j].y - p_y;
		float dist = sqrt(dx * dx + dy * dy);

		if ( dist != 0 )
		{
			float force = ELEM(edge_matrix, n, i, j)
				? K_s * (L - dist) / dist
				: K_r / (dist * dist * dist);

			v_x -= force * dx;
			v_y -= force * dy;
		}
	}
	__syncthreads();

	// adjust velocity to not exceed a certain magnitude
	float v_magnitude_sqr = v_x * v_x + v_y * v_y;

	if ( v_magnitude_sqr > MAX_VELOCITY_MAGNITUDE_SQR )
	{
		v_x *= sqrt(MAX_VELOCITY_MAGNITUDE_SQR / v_magnitude_sqr);
		v_y *= sqrt(MAX_VELOCITY_MAGNITUDE_SQR / v_magnitude_sqr);
	}

	positions[i].x += v_x;
	positions[i].y += v_y;
	velocities[i].x = 0.1f * v_x;
	velocities[i].y = 0.1f * v_y;
}



__global__
void fdl_kernel_3d(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= n )
	{
		return;
	}

	// define force constants
	const float K_r = 0.2f;
	const float K_s = 1.0f;
	const float L = 2.2f;

	// get node position
	float p_x = positions[i].x;
	float p_y = positions[i].y;
	float p_z = positions[i].y;

	// compute node velocity
	float v_x = velocities[i].x;
	float v_y = velocities[i].y;
	float v_z = velocities[i].y;

	for ( int j = 0; j < n; j++ )
	{
		float dx = positions[j].x - p_x;
		float dy = positions[j].y - p_y;
		float dz = positions[j].z - p_z;
		float dist = sqrt(dx * dx + dy * dy + dz * dz);

		if ( dist != 0 )
		{
			float force = ELEM(edge_matrix, n, i, j)
				? K_s * (L - dist) / dist
				: K_r / (dist * dist * dist);

			v_x -= force * dx;
			v_y -= force * dy;
			v_z -= force * dz;
		}
	}
	__syncthreads();

	// adjust velocity to not exceed a certain magnitude
	float v_magnitude_sqr = v_x * v_x + v_y * v_y + v_z * v_z;

	if ( v_magnitude_sqr > MAX_VELOCITY_MAGNITUDE_SQR )
	{
		v_x *= sqrt(MAX_VELOCITY_MAGNITUDE_SQR / v_magnitude_sqr);
		v_y *= sqrt(MAX_VELOCITY_MAGNITUDE_SQR / v_magnitude_sqr);
		v_z *= sqrt(MAX_VELOCITY_MAGNITUDE_SQR / v_magnitude_sqr);
	}

	positions[i].x += v_x;
	positions[i].y += v_y;
	positions[i].z += v_z;
	velocities[i].x = 0.1f * v_x;
	velocities[i].y = 0.1f * v_y;
	velocities[i].z = 0.1f * v_z;
}



void fdl_2d_gpu(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix)
{
	const int BLOCK_SIZE = 256;
	const int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

	fdl_kernel_2d<<<GRID_SIZE, BLOCK_SIZE>>>(
		n,
		positions,
		velocities,
		edge_matrix);
	CUDA_SAFE_CALL(cudaGetLastError());
}



void fdl_3d_gpu(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix)
{
	const int BLOCK_SIZE = 256;
	const int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

	fdl_kernel_3d<<<GRID_SIZE, BLOCK_SIZE>>>(
		n,
		positions,
		velocities,
		edge_matrix);
	CUDA_SAFE_CALL(cudaGetLastError());
}

#include "fdl.h"
#include <cmath>
#include <cstdint>



const float MAX_VELOCITY_MAGNITUDE_SQR = 10.0f;



#define ELEM(data, cols, i, j) (data)[(int64_t)(i) * (cols) + (j)]



void fdl_2d_cpu(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix)
{
	// define force constants
	const float K_r = 0.2f;
	const float K_s = 1.0f;
	const float L = 2.2f;

	// update each node
	for ( int i = 0; i < n; i++ )
	{
		// compute node velocity
		float v_x = velocities[i].x;
		float v_y = velocities[i].y;

		for ( int j = 0; j < n; j++ )
		{
			float dx = positions[j].x - positions[i].x;
			float dy = positions[j].y - positions[i].y;
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

		velocities[i].x = v_x;
		velocities[i].y = v_y;
	}

	for ( int i = 0; i < n; i++ )
	{
		// adjust velocity to not exceed a certain magnitude
		float v_x = velocities[i].x;
		float v_y = velocities[i].y;
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
}



void fdl_3d_cpu(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix)
{
	// define force constants
	const float K_r = 0.2f;
	const float K_s = 1.0f;
	const float L = 2.2f;

	// update each node
	for ( int i = 0; i < n; i++ )
	{
		// compute node velocity
		float v_x = velocities[i].x;
		float v_y = velocities[i].y;
		float v_z = velocities[i].y;

		for ( int j = 0; j < n; j++ )
		{
			float dx = positions[j].x - positions[i].x;
			float dy = positions[j].y - positions[i].y;
			float dz = positions[j].z - positions[i].z;
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
		velocities[i].x = v_x;
		velocities[i].y = v_y;
		velocities[i].z = v_z;
	}

	for ( int i = 0; i < n; i++ )
	{
		// adjust velocity to not exceed a certain magnitude
		float v_x = velocities[i].x;
		float v_y = velocities[i].y;
		float v_z = velocities[i].z;
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
}

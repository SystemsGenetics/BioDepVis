#include <cmath>
#include "fdl.h"



const float MAX_DISPLACEMENT_SQR = 10.0f;



#define ELEM(data, cols, i, j) (data)[(i) * (cols) + (j)]



void fdl_2d_cpu(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix)
{
	const float K_r = 0.2f;
	const float K_s = 1.0f;
	const float L = 2.2f;

	for ( int i = 0; i < n; i++ )
	{
		for ( int j = 0; j < n; j++ )
		{
			if ( i == j )
			{
				continue;
			}

			float dx = positions[j].x - positions[i].x;
			float dy = positions[j].y - positions[i].y;
			float dist = sqrt(dx * dx + dy * dy);

			if ( dist != 0 )
			{
				float force = ELEM(edge_matrix, n, i, j)
					? K_s * (L - dist) / dist
					: K_r / (dist * dist * dist);

				velocities[i].x -= force * dx;
				velocities[i].y -= force * dy;

				velocities[j].x += force * dx;
				velocities[j].y += force * dy;
			}
		}

		float dx = velocities[i].x;
		float dy = velocities[i].y;
		float disp_sqr = dx * dx + dy * dy;

		if ( disp_sqr > MAX_DISPLACEMENT_SQR )
		{
			dx *= sqrt(MAX_DISPLACEMENT_SQR / disp_sqr);
			dy *= sqrt(MAX_DISPLACEMENT_SQR / disp_sqr);
		}

		positions[i].x += dx;
		positions[i].y += dy;
		velocities[i].x *= 0.1f;
		velocities[i].y *= 0.1f;
	}
}



void fdl_3d_cpu(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix)
{
	const float K_r = 0.2f;
	const float K_s = 1.0f;
	const float L = 2.2f;

	for ( int i = 0; i < n; i++ )
	{
		for ( int j = 0; j < n; j++ )
		{
			if ( i == j )
			{
				continue;
			}

			float dx = positions[j].x - positions[i].x;
			float dy = positions[j].y - positions[i].y;
			float dz = positions[j].z - positions[i].z;
			float dist = sqrt(dx * dx + dy * dy + dz * dz);

			if ( dist != 0 )
			{
				float force = ELEM(edge_matrix, n, i, j)
					? K_s * (L - dist) / dist
					: K_r / (dist * dist * dist);

				velocities[i].x -= force * dx;
				velocities[i].y -= force * dy;
				velocities[i].z -= force * dz;

				velocities[j].x += force * dx;
				velocities[j].y += force * dy;
				velocities[j].z += force * dz;
			}
		}

		float dx = velocities[i].x;
		float dy = velocities[i].y;
		float dz = velocities[i].z;
		float disp_sqr = dx * dx + dy * dy + dz * dz;

		if ( disp_sqr > MAX_DISPLACEMENT_SQR )
		{
			dx *= sqrt(MAX_DISPLACEMENT_SQR / disp_sqr);
			dy *= sqrt(MAX_DISPLACEMENT_SQR / disp_sqr);
			dz *= sqrt(MAX_DISPLACEMENT_SQR / disp_sqr);
		}

		positions[i].x += dx;
		positions[i].y += dy;
		positions[i].z += dz;
		velocities[i].x *= 0.1f;
		velocities[i].y *= 0.1f;
		velocities[i].z *= 0.1f;
	}
}

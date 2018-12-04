#include "fdl.h"
#include <cmath>
#include <cstdint>



const float MAX_VELOCITY_MAGNITUDE_SQR = 10.0f;



#define ELEM(data, cols, i, j) (data)[(int64_t)(i) * (cols) + (j)]



/**
 * Perform one iteration of 2D force-directed layout on a graph.
 *
 * @param n
 * @param positions
 * @param velocities
 * @param edge_matrix
 */
void fdl_2d_cpu(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix)
{
	// define force constants
	const float K_r = 0.2f;
	const float K_s = 1.0f;
	const float L = 2.2f;

	// compute velocity of each node
	for ( int i = 0; i < n; i++ )
	{
		// apply force from each node in the graph
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

		// adjust velocity to not exceed a certain magnitude
		float v_magnitude_sqr = v_x * v_x + v_y * v_y;

		if ( v_magnitude_sqr > MAX_VELOCITY_MAGNITUDE_SQR )
		{
			v_x *= sqrt(MAX_VELOCITY_MAGNITUDE_SQR / v_magnitude_sqr);
			v_y *= sqrt(MAX_VELOCITY_MAGNITUDE_SQR / v_magnitude_sqr);
		}

		// save velocity
		velocities[i].x = v_x;
		velocities[i].y = v_y;
	}

	// update position of each node
	for ( int i = 0; i < n; i++ )
	{
		positions[i].x += velocities[i].x;
		positions[i].y += velocities[i].y;
		velocities[i].x *= 0.1f;
		velocities[i].y *= 0.1f;
	}
}



/**
 * Perform one iteration of 3D force-directed layout on a graph.
 *
 * @param n
 * @param positions
 * @param velocities
 * @param edge_matrix
 */
void fdl_3d_cpu(int n, Vector3 *positions, Vector3 *velocities, const bool *edge_matrix)
{
	// define force constants
	const float K_r = 0.2f;
	const float K_s = 1.0f;
	const float L = 2.2f;

	// compute velocity of each node
	for ( int i = 0; i < n; i++ )
	{
		// apply force from each node in the graph
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

		// adjust velocity to not exceed a certain magnitude
		float v_magnitude_sqr = v_x * v_x + v_y * v_y + v_z * v_z;

		if ( v_magnitude_sqr > MAX_VELOCITY_MAGNITUDE_SQR )
		{
			v_x *= sqrt(MAX_VELOCITY_MAGNITUDE_SQR / v_magnitude_sqr);
			v_y *= sqrt(MAX_VELOCITY_MAGNITUDE_SQR / v_magnitude_sqr);
			v_z *= sqrt(MAX_VELOCITY_MAGNITUDE_SQR / v_magnitude_sqr);
		}

		// save velocity
		velocities[i].x = v_x;
		velocities[i].y = v_y;
		velocities[i].z = v_z;
	}

	// update position of each node
	for ( int i = 0; i < n; i++ )
	{
		positions[i].x += velocities[i].x;
		positions[i].y += velocities[i].y;
		positions[i].z += velocities[i].z;
		velocities[i].x *= 0.1f;
		velocities[i].y *= 0.1f;
		velocities[i].z *= 0.1f;
	}
}

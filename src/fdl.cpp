#include <cmath>
#include "fdl.h"

const float MAX_DISPLACEMENT_SQR = 10.0f;

#define ELEM(data, cols, i, j) (data)[(i) * (cols) + (j)]

void fdl_2d_cpu(int n, vec3_t *positions, vec3_t *positions_d, const bool *edge_matrix)
{
    const float K_r = 0.2f;
    const float K_s = 1.0f;
    const float L = 2.2f;

    for ( int i = 0; i < n; i++ ) {
        for ( int j = 0; j < n; j++ ) {
            if ( i == j ) {
                continue;
            }

            float dx = positions[j].x - positions[i].x;
            float dy = positions[j].y - positions[i].y;
            float dist = sqrtf(dx * dx + dy * dy);

            if ( dist != 0 ) {
                float force = ELEM(edge_matrix, n, i, j)
                    ? K_s * (L - dist) / dist
                    : K_r / (dist * dist * dist);

                positions_d[i].x -= force * dx;
                positions_d[i].y -= force * dy;

                positions_d[j].x += force * dx;
                positions_d[j].y += force * dy;
            }
        }

        float dx = positions_d[i].x;
        float dy = positions_d[i].y;
        float disp_sqr = dx * dx + dy * dy;

        if ( disp_sqr > MAX_DISPLACEMENT_SQR ) {
            dx *= sqrtf(MAX_DISPLACEMENT_SQR / disp_sqr);
            dy *= sqrtf(MAX_DISPLACEMENT_SQR / disp_sqr);
        }

        positions[i].x += dx;
        positions[i].y += dy;
        positions_d[i].x *= 0.1f;
        positions_d[i].y *= 0.1f;
    }
}

void fdl_3d_cpu(int n, vec3_t *positions, vec3_t *positions_d, const bool *edge_matrix)
{
    const float K_r = 0.2f;
    const float K_s = 1.0f;
    const float L = 2.2f;

    for ( int i = 0; i < n; i++ ) {
        for ( int j = 0; j < n; j++ ) {
            if ( i == j ) {
                continue;
            }

            float dx = positions[j].x - positions[i].x;
            float dy = positions[j].y - positions[i].y;
            float dz = positions[j].z - positions[i].z;
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);

            if ( dist != 0 ) {
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

        float dx = positions_d[i].x;
        float dy = positions_d[i].y;
        float dz = positions_d[i].z;
        float disp_sqr = dx * dx + dy * dy + dz * dz;

        if ( disp_sqr > MAX_DISPLACEMENT_SQR ) {
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

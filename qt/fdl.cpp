#include <cassert>
#include <cmath>
#include "fdl.h"

const float MAX_DISPLACEMENT_SQR = 10.0f;

inline int ELEM(int *data, int cols, int i, int j)
{
    return data[i * cols + j];
}

void fdl_2d_cpu(int n, vec3_t *coords, vec3_t *coords_d, int *edge_matrix)
{
    float K_r = 25.0f;
    float K_s = 15.0f;
    float L = 1.2f;
    float dt = 0.004f;

    for ( int i = 0; i < n; i++ ) {
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

        assert(!std::isnan(dx) && !std::isnan(dy));

        coords[i].x += dx;
        coords[i].y += dy;
        coords_d[i].x *= 0.09f;
        coords_d[i].y *= 0.09f;
    }
}

void fdl_3d_cpu(int n, vec3_t *coords, vec3_t *coords_d, int *edge_matrix)
{
    float K_r = 25.0f;
    float K_s = 15.0f;
    float L = 1.2f;
    float dt = 0.004f;

    for ( int i = 0; i < n; i++ ) {
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

        assert(!std::isnan(dx) && !std::isnan(dy) && !std::isnan(dz));

        coords[i].x += dx;
        coords[i].y += dy;
        coords[i].z += dz;
        coords_d[i].x *= 0.09f;
        coords_d[i].y *= 0.09f;
        coords_d[i].z *= 0.09f;
    }
}

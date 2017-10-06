#include <cassert>
#include <cmath>
#include "fdl.h"

const float MAX_DIST_SQR = 10.0f;

void force_directed_layout(Graph *g)
{
    QVector<vec3_t>& coords = g->coords();
    QVector<vec3_t>& coords_d = g->coords_d();
    const Matrix& edge_matrix = g->edge_matrix();

    float K_r = 25.0f;
    float K_s = 15.0f;
    float L = 1.2f;
    float dt = 0.004f;

    for ( int i = 0; i < coords.size(); i++ ) {
        for ( int j = 0; j < coords.size(); j++ ) {
            if ( i == j ) {
                continue;
            }

            float dx = coords[j].x - coords[i].x;
            float dy = coords[j].y - coords[i].y;
            float dist = sqrtf(dx * dx + dy * dy);

            if ( dist != 0 ) {
                float force = (edge_matrix.elem(i, j) != 0)
                    ? K_s * (L - dist)
                    : K_r / (dist * dist);

                coords_d[i].x -= force * dx / dist;
                coords_d[i].y -= force * dy / dist;

                coords_d[j].x += force * dx / dist;
                coords_d[j].y += force * dy / dist;
            }
        }

        float dx = coords_d[i].x * dt;
        float dy = coords_d[i].y * dt;
        float dist_sqr = dx * dx + dy * dy;

        if ( dist_sqr > MAX_DIST_SQR ) {
            dx *= sqrtf(MAX_DIST_SQR / dist_sqr);
            dy *= sqrtf(MAX_DIST_SQR / dist_sqr);
        }

        assert(!std::isnan(dx) && !std::isnan(dy));

        coords[i].x += dx;
        coords[i].y += dy;
        coords_d[i].x *= 0.09f;
        coords_d[i].y *= 0.09f;
    }
}

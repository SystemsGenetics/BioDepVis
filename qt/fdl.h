#ifndef FDL_H
#define FDL_H

#include "vector.h"

void force_directed_layout_2d(int n, vec3_t *coords, vec3_t *coords_d, int *edge_matrix);
void force_directed_layout_3d(int n, vec3_t *coords, vec3_t *coords_d, int *edge_matrix);

#endif

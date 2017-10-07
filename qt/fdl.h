#ifndef FDL_H
#define FDL_H

#include "vector.h"

void fdl_2d_cpu(int n, vec3_t *coords, vec3_t *coords_d, int *edge_matrix);
void fdl_3d_cpu(int n, vec3_t *coords, vec3_t *coords_d, int *edge_matrix);
void fdl_2d_gpu(int n, vec3_t *coords, vec3_t *coords_d, int *edge_matrix);
void fdl_3d_gpu(int n, vec3_t *coords, vec3_t *coords_d, int *edge_matrix);

void * gpu_malloc(int size);
void gpu_free(void *ptr);
void gpu_read(void *dst, void *src, int size);
void gpu_write(void *dst, void *src, int size);

#endif

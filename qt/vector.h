#ifndef VECTOR_H
#define VECTOR_H

typedef struct {
	float x;
	float y;
	float z;
} vec3_t;

vec3_t vector_cross(const vec3_t& v1, const vec3_t& v2);
float vector_norm(const vec3_t& v);
vec3_t vector_normalize(const vec3_t& v);

vec3_t operator-(const vec3_t& v);
vec3_t operator+(const vec3_t& v1, const vec3_t& v2);
vec3_t operator-(const vec3_t& v1, const vec3_t& v2);
float operator*(const vec3_t& v1, const vec3_t& v2);
vec3_t operator*(const vec3_t& v, float c);
vec3_t operator*(float c, const vec3_t& v);
vec3_t operator/(const vec3_t& v, float c);

#endif

#include <cmath>
#include "vector.h"

vec3_t vector_cross(const vec3_t& v1, const vec3_t& v2)
{
	return (vec3_t) {
		v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x
	};
}

float vector_norm(const vec3_t& v)
{
	return sqrtf(v * v);
}

vec3_t vector_normalize(const vec3_t& v)
{
	return v / vector_norm(v);
}

vec3_t operator-(const vec3_t& v)
{
	return (vec3_t) { -v.x, -v.y, -v.z };
}

vec3_t operator+(const vec3_t& v1, const vec3_t& v2)
{
	return (vec3_t) { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
}

vec3_t operator-(const vec3_t& v1, const vec3_t& v2)
{
	return (vec3_t) { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
}

float operator*(const vec3_t& v1, const vec3_t& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

vec3_t operator*(const vec3_t& v, float c)
{
	return (vec3_t) { v.x * c, v.y * c, v.z * c };
}

vec3_t operator*(float c, const vec3_t& v)
{
	return v * c;
}

vec3_t operator/(const vec3_t& v, float c)
{
	return (vec3_t) { v.x / c, v.y / c, v.z / c };
}

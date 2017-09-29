#ifndef CAMERA_H
#define CAMERA_H

#include "vector.h"

class Camera {
private:
	vec3_t _default_pos;
	vec3_t _default_aim;
	vec3_t _default_up;
	float _default_elev;
	float _default_azim;

	vec3_t _pos;
	vec3_t _aim;
	vec3_t _up;
	float _elev;
	float _azim;

	float _near_plane;
	float _far_plane;
	float _fov;
	bool _inverted;
	float _translate_x;
	float _translate_y;
	float _zoom;

public:
	Camera(const vec3_t& pos, const vec3_t& aim, const vec3_t& up);
	Camera() {};

	vec3_t& pos() { return this->_pos; }
	vec3_t& aim() { return this->_aim; }
	vec3_t& up() { return this->_up; }

	void display(int w, int h);
	void reset();

private:
	void compute_coords(const vec3_t& pos, const vec3_t& aim, const vec3_t& up);
};

#endif

#include <cmath>
#include <GL/glu.h>
#include <QDebug>
#include "camera.h"

const float EPSILON = 0.0001f;

#define deg_to_rad(x) ((x) * M_PI / 180.0f)
#define rad_to_deg(x) ((x) * 180.0f / M_PI)

Camera::Camera(const vec3_t& pos, const vec3_t& aim, const vec3_t& up)
{
	this->compute_coords(pos, aim, up);

	// compute the angle around the x axis
	vec3_t update_pos = pos - aim;
	vec3_t axis_origin = { update_pos.x, 0, 0 };
	float dist = vector_norm(axis_origin - update_pos);

	vec3_t temp1 = vector_normalize(update_pos);
	vec3_t temp2 = vector_normalize((vec3_t) { update_pos.x, 0, dist });

	this->_elev = rad_to_deg(acosf(temp1 * temp2));

	// compute the angle around the y axis
	axis_origin = (vec3_t) { 0, update_pos.y, 0 };
	dist = vector_norm(axis_origin - update_pos);

	vec3_t temp3 = vector_normalize((vec3_t) { 0, update_pos.y, dist });

	this->_azim = 360.0f - rad_to_deg(acosf(temp1 * temp3));

	this->_near_plane = 0.1f;
	this->_far_plane = 1000.0f;
	this->_fov = 60.0f;
	this->_inverted = false;
	this->_translate_x = 0.0f;
	this->_translate_y = 0.0f;
	this->_zoom = 0.0f;

	this->_default_pos = this->_pos;
	this->_default_aim = this->_aim;
	this->_default_up = this->_up;
	this->_default_elev = this->_elev;
	this->_default_azim = this->_azim;
}

void Camera::compute_coords(const vec3_t& pos, const vec3_t& aim, const vec3_t& up)
{
	vec3_t zaxis = pos - aim;

	if ( vector_norm(zaxis) < EPSILON ) {
		qWarning("warning: camera position and aim position are coincident");
	}

	vec3_t dir = -vector_normalize(zaxis);
	vec3_t up1 = vector_normalize(up);
	vec3_t xaxis = vector_cross(dir, up1);

	if ( vector_norm(xaxis) < EPSILON ) {
		qWarning("warning: up vector is parallel to aim");
	}

	this->_pos = pos;
	this->_aim = aim;
	this->_up = vector_cross(vector_normalize(xaxis), dir);
}

void Camera::display(int w, int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	_near_plane = 0.0001f;
	_fov = fmin(fmax(30.0f + _zoom, 1.0f), 270.0f);

	gluPerspective(_fov, (float)w / h, _near_plane, _far_plane);
	gluLookAt(
		_pos.x, _pos.y, _pos.z,
		_aim.x, _aim.y, _aim.z,
		_up.x, _up.y, _up.z
	);

	glTranslatef(_translate_x, -_translate_y, 0.0f);
	glRotatef(_elev, 1, 0, 0);
	glRotatef(_azim, 0, 1, 0);
}

void Camera::reset()
{
	this->_pos = this->_default_pos;
	this->_aim = this->_default_aim;
	this->_up = this->_default_up;
	this->_elev = this->_default_elev;
	this->_azim = this->_default_azim;
}

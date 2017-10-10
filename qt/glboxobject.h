#ifndef GLBOXOBJECT_H
#define GLBOXOBJECT_H

#include <QOpenGLBuffer>
#include <QVector>
#include "database.h"
#include "globject.h"

class GLBoxObject : public GLObject {
private:
	QVector<vec3_t> _positions;
	QVector<color_t> _colors;
	QOpenGLBuffer _vbo_positions;
	QOpenGLBuffer _vbo_colors;

public:
	GLBoxObject() {};
	~GLBoxObject();

	void initialize() override;
	void paint() override;

	void update(Database *db, const QVector<node_ref_t>& nodes);

private:
	void create_box(const vec3_t& C, float W, const color_t& color);
};

#endif

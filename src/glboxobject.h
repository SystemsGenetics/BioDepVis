#ifndef GLBOXOBJECT_H
#define GLBOXOBJECT_H

#include <QOpenGLBuffer>
#include <QVector>
#include "database.h"
#include "globject.h"

class GLBoxObject : public GLObject {
private:
	Database *_db;
	QVector<node_ref_t> _nodes;
	QVector<float> _sizes;
	QVector<vec3_t> _positions;
	QVector<color_t> _colors;
	QOpenGLBuffer _vbo_positions;
	QOpenGLBuffer _vbo_colors;

public:
	GLBoxObject(Database *db);
	~GLBoxObject();

	void initialize() override;
	void paint() override;

	void append(const QVector<node_ref_t>& nodes, float L, const color_t& color);
	void clear();
	void update();

private:
	void update_cube(int i, const vec3_t& C, float L);
};

#endif

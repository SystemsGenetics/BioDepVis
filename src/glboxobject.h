#ifndef GLBOXOBJECT_H
#define GLBOXOBJECT_H

#include <QOpenGLBuffer>
#include <QVector>
#include "database.h"
#include "globject.h"



class GLBoxObject : public GLObject
{
private:
	Database *_db;
	QVector<NodeRef> _nodes;
	QVector<float> _sizes;
	QVector<Vector3> _positions;
	QVector<Color> _colors;
	QOpenGLBuffer _vbo_positions;
	QOpenGLBuffer _vbo_colors;

public:
	GLBoxObject(Database *db);
	~GLBoxObject();

	void initialize() override;
	void paint() override;

	void append(const QVector<NodeRef>& nodes, float L, const Color& color);
	void clear();
	void update();

private:
	void update_cube(int i, const Vector3& C, float L);
};



#endif

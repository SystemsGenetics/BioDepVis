#ifndef GLALIGNOBJECT_H
#define GLALIGNOBJECT_H

#include <QOpenGLBuffer>
#include <QVector>
#include "alignment.h"
#include "globject.h"



class GLAlignObject : public GLObject
{
private:
	Alignment *_align;
	QVector<color_t> _edge_colors;
	QOpenGLBuffer _vbo_positions;
	QOpenGLBuffer _vbo_colors;

public:
	GLAlignObject(Alignment *align);
	~GLAlignObject();

	void initialize() override;
	void paint() override;
};



#endif

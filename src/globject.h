#ifndef GLOBJECT_H
#define GLOBJECT_H

#include <QOpenGLVertexArrayObject>

class GLObject {
protected:
	QOpenGLVertexArrayObject _vao;

public:
	virtual ~GLObject() {};

	virtual void initialize() = 0;
	virtual void paint() = 0;
};

#endif

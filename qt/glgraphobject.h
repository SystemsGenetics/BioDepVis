#ifndef GLGRAPHOBJECT_H
#define GLGRAPHOBJECT_H

#include <QOpenGLBuffer>
#include <QVector>
#include "globject.h"
#include "graph.h"

class GLGraphObject : public GLObject {
private:
	Graph *_graph;
	QVector<color_t> _node_colors;
	QVector<color_t> _edge_colors;
	QOpenGLBuffer _vbo_coords;
	QOpenGLBuffer _vbo_colors;

public:
	GLGraphObject(Graph *graph);
	~GLGraphObject();

	void initialize() override;
	void paint() override { paint(false); };
	void paint(bool show_modules);
};

#endif

#include <QOpenGLFunctions>
#include "glgraphobject.h"

const color_t EDGE_COLORS[] = {
	{ 0.65f, 0.81f, 0.89f, 0.30f },
	{ 0.50f, 0.50f, 0.78f, 0.30f },
	{ 0.42f, 0.24f, 0.60f, 0.30f },
	{ 0.12f, 0.47f, 0.71f, 0.30f },
	{ 0.70f, 0.87f, 0.54f, 0.30f },
	{ 1.00f, 0.50f, 0.00f, 0.30f }
};
const int NUM_EDGE_COLORS = sizeof(EDGE_COLORS) / sizeof(color_t);

GLGraphObject::GLGraphObject(Graph *graph)
{
	_graph = graph;
}

GLGraphObject::~GLGraphObject()
{
	_vbo_coords.destroy();
	_vbo_colors.destroy();
}

void GLGraphObject::initialize()
{
	QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();
	static int color_idx = 0;

	// initialize node colors
	_node_colors.reserve(_graph->nodes().size());
	for ( int i = 0; i < _graph->nodes().size(); i++ ) {
		_node_colors.push_back(color_t { 0, 0, 0, 1 });
	}

	// initialize edge colors
	_edge_colors.reserve(_graph->edges().size());
	for ( int i = 0; i < _graph->edges().size(); i++ ) {
		_edge_colors.push_back(EDGE_COLORS[color_idx]);
	}

	color_idx = (color_idx + 1) % NUM_EDGE_COLORS;

	// initialize vertex array object
	_vao.create();

	QOpenGLVertexArrayObject::Binder vaoBinder(&_vao);

	// initialize position buffer
	_vbo_coords.create();
	_vbo_coords.bind();
	_vbo_coords.allocate(_graph->nodes().size() * sizeof(vec3_t));

	f->glEnableVertexAttribArray(0);
	f->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	_vbo_coords.release();

	// initialize color buffer
	int num_colors = qMax(_graph->nodes().size(), _graph->edges().size());

	_vbo_colors.create();
	_vbo_colors.bind();
	_vbo_colors.allocate(num_colors * sizeof(color_t));

	f->glEnableVertexAttribArray(1);
	f->glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	_vbo_colors.release();
}

void GLGraphObject::paint(bool show_modules)
{
	QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();
	QOpenGLVertexArrayObject::Binder vaoBinder(&_vao);

	// write node positions
	_vbo_coords.bind();
	_vbo_coords.write(0, _graph->coords().data(), _graph->nodes().size() * sizeof(vec3_t));
	_vbo_coords.release();

	// write node colors
	const color_t *color_data = show_modules
		? _graph->colors().data()
		: _node_colors.data();

	_vbo_colors.bind();
	_vbo_colors.write(0, color_data, _graph->nodes().size() * sizeof(color_t));
	_vbo_colors.release();

	// draw nodes
	f->glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	f->glDrawArrays(GL_POINTS, 0, _graph->coords().size());

	// write edge colors
	_vbo_colors.bind();
	_vbo_colors.write(0, _edge_colors.data(), _graph->edges().size() * sizeof(color_t));
	_vbo_colors.release();

	// draw edges
	f->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	f->glLineWidth(0.001f);
	f->glDrawElements(GL_LINES, _graph->edges().size(), GL_UNSIGNED_INT, _graph->edges().data());
}
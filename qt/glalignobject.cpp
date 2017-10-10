#include <QOpenGLFunctions>
#include "glalignobject.h"

const color_t EDGE_COLORS[] = {
	{ 0.70f, 0.19f, 0.29f, 0.10f },
	{ 0.89f, 0.50f, 0.79f, 0.10f },
	{ 0.89f, 0.50f, 0.79f, 0.10f }
};
const int NUM_EDGE_COLORS = sizeof(EDGE_COLORS) / sizeof(color_t);

GLAlignObject::GLAlignObject(Alignment *align)
{
	_align = align;
}

GLAlignObject::~GLAlignObject()
{
	_vbo_positions.destroy();
	_vbo_colors.destroy();
}

void GLAlignObject::initialize()
{
	QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();
	static int color_idx = 0;

	// initialize edge colors
	_edge_colors.reserve(_align->edges().size());
	for ( int i = 0; i < _align->edges().size(); i++ ) {
		_edge_colors.push_back(EDGE_COLORS[color_idx]);
	}

	color_idx = (color_idx + 1) % NUM_EDGE_COLORS;

	// initialize vertex array object
	_vao.create();

	QOpenGLVertexArrayObject::Binder vaoBinder(&_vao);

	// initialize position buffer
	_vbo_positions.create();
	_vbo_positions.bind();
	_vbo_positions.allocate(_align->edges().size() * sizeof(align_edge_t));

	f->glEnableVertexAttribArray(0);
	f->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	_vbo_positions.release();

	// initialize color buffer
	_vbo_colors.create();
	_vbo_colors.bind();
	_vbo_colors.allocate(_edge_colors.data(), _align->edges().size() * sizeof(color_t));

	f->glEnableVertexAttribArray(1);
	f->glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	_vbo_colors.release();
}

void GLAlignObject::paint()
{
	QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();
	QOpenGLVertexArrayObject::Binder vaoBinder(&_vao);

	// write edge positions
	_vbo_positions.bind();
	_vbo_positions.write(0, _align->vertices().data(), _align->edges().size() * sizeof(align_edge_t));
	_vbo_positions.release();

	// draw edges
	f->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	f->glLineWidth(0.1f);
	f->glDrawArrays(GL_LINES, 0, _align->edges().size());
}

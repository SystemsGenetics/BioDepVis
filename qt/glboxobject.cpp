#include <QDebug>
#include <QOpenGLFunctions>
#include "glboxobject.h"

const color_t BOX_COLOR = { 1, 0, 0, 1 };
const float BOX_WIDTH = 5.0f;
const int VERTICES_PER_BOX = 24;

GLBoxObject::~GLBoxObject()
{
	_vbo_positions.destroy();
	_vbo_colors.destroy();
}

void GLBoxObject::initialize()
{
	QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();

	// initialize vertex array object
	_vao.create();

	QOpenGLVertexArrayObject::Binder vaoBinder(&_vao);

	// initialize position buffer
	_vbo_positions.create();
	_vbo_positions.bind();

	f->glEnableVertexAttribArray(0);
	f->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	_vbo_positions.release();

	// initialize color buffer
	_vbo_colors.create();
	_vbo_colors.bind();

	f->glEnableVertexAttribArray(1);
	f->glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	_vbo_colors.release();
}

void GLBoxObject::paint()
{
	QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();
	QOpenGLVertexArrayObject::Binder vaoBinder(&_vao);

	// draw boxes
	f->glBlendFunc(GL_ONE, GL_ZERO);
	f->glLineWidth(1.0f);
	f->glDrawArrays(GL_LINES, 0, _positions.size());
}

void GLBoxObject::update(Database *db, const QVector<node_ref_t>& nodes)
{
	qDebug() << nodes.size();

	// create boxes
	int num_positions = nodes.size() * VERTICES_PER_BOX;

	_positions.clear();
	_positions.reserve(num_positions);

	_colors.clear();
	_colors.reserve(num_positions);

	for ( const node_ref_t& ref : nodes ) {
		const vec3_t& center = db->graphs()[ref.graph_id]->coords()[ref.node_id];

		create_box(center, BOX_WIDTH, BOX_COLOR);
	}

	QOpenGLVertexArrayObject::Binder vaoBinder(&_vao);

	// allocate position buffer
	_vbo_positions.bind();
	_vbo_positions.allocate(_positions.data(), num_positions * sizeof(vec3_t));
	_vbo_positions.release();

	// allocate color buffer
	_vbo_colors.bind();
	_vbo_colors.allocate(_colors.data(), num_positions * sizeof(color_t));
	_vbo_colors.release();
}

void GLBoxObject::create_box(const vec3_t& C, float W, const color_t& color)
{
	qDebug() << C.x << C.y << C.z;

	// append positions
	_positions.push_back(vec3_t { C.x - W/2, C.y + W/2, C.z - W/2 });
	_positions.push_back(vec3_t { C.x + W/2, C.y + W/2, C.z - W/2 });
	_positions.push_back(vec3_t { C.x + W/2, C.y + W/2, C.z - W/2 });
	_positions.push_back(vec3_t { C.x + W/2, C.y - W/2, C.z - W/2 });
	_positions.push_back(vec3_t { C.x + W/2, C.y - W/2, C.z - W/2 });
	_positions.push_back(vec3_t { C.x - W/2, C.y - W/2, C.z - W/2 });
	_positions.push_back(vec3_t { C.x - W/2, C.y - W/2, C.z - W/2 });
	_positions.push_back(vec3_t { C.x - W/2, C.y + W/2, C.z - W/2 });

	_positions.push_back(vec3_t { C.x - W/2, C.y + W/2, C.z + W/2 });
	_positions.push_back(vec3_t { C.x + W/2, C.y + W/2, C.z + W/2 });
	_positions.push_back(vec3_t { C.x + W/2, C.y + W/2, C.z + W/2 });
	_positions.push_back(vec3_t { C.x + W/2, C.y - W/2, C.z + W/2 });
	_positions.push_back(vec3_t { C.x + W/2, C.y - W/2, C.z + W/2 });
	_positions.push_back(vec3_t { C.x - W/2, C.y - W/2, C.z + W/2 });
	_positions.push_back(vec3_t { C.x - W/2, C.y - W/2, C.z + W/2 });
	_positions.push_back(vec3_t { C.x - W/2, C.y + W/2, C.z + W/2 });

	_positions.push_back(vec3_t { C.x - W/2, C.y + W/2, C.z - W/2 });
	_positions.push_back(vec3_t { C.x - W/2, C.y + W/2, C.z + W/2 });
	_positions.push_back(vec3_t { C.x + W/2, C.y + W/2, C.z - W/2 });
	_positions.push_back(vec3_t { C.x + W/2, C.y + W/2, C.z + W/2 });
	_positions.push_back(vec3_t { C.x + W/2, C.y - W/2, C.z - W/2 });
	_positions.push_back(vec3_t { C.x + W/2, C.y - W/2, C.z + W/2 });
	_positions.push_back(vec3_t { C.x - W/2, C.y - W/2, C.z - W/2 });
	_positions.push_back(vec3_t { C.x - W/2, C.y - W/2, C.z + W/2 });

	// append colors
	for ( int i = 0; i < VERTICES_PER_BOX; i++ ) {
		_colors.push_back(color);
	}
}

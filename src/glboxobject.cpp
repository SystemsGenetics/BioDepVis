#include <QOpenGLFunctions>
#include "glboxobject.h"



const int VERTICES_PER_BOX = 24;



GLBoxObject::GLBoxObject(Database *db):
	_db(db)
{
}



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

	// write positions
	_vbo_positions.bind();
	_vbo_positions.allocate(_positions.data(), _positions.size() * sizeof(vec3_t));
	_vbo_positions.release();

	// draw boxes
	f->glBlendFunc(GL_ONE, GL_ZERO);
	f->glLineWidth(1.0f);
	f->glDrawArrays(GL_LINES, 0, _positions.size());
}



void GLBoxObject::append(const QVector<node_ref_t>& nodes, float L, const color_t& color)
{
	// update nodes and sizes
	_nodes.append(nodes);

	_sizes.reserve(_nodes.size());
	for ( int i = 0; i < _nodes.size(); i++ )
	{
		_sizes.push_back(L);
	}

	// allocate positions for boxes
	int num_positions = _nodes.size() * VERTICES_PER_BOX;

	_positions.resize(num_positions);

	// allocate, initialize colors for boxes
	_colors.reserve(num_positions);

	for ( int i = 0; i < nodes.size() * VERTICES_PER_BOX; i++ )
	{
		_colors.push_back(color);
	}

	QOpenGLVertexArrayObject::Binder vaoBinder(&_vao);

	// allocate position buffer
	_vbo_positions.bind();
	_vbo_positions.allocate(_positions.size() * sizeof(vec3_t));
	_vbo_positions.release();

	// allocate, update color buffer
	_vbo_colors.bind();
	_vbo_colors.allocate(_colors.data(), _colors.size() * sizeof(color_t));
	_vbo_colors.release();
}



void GLBoxObject::clear()
{
	_nodes.clear();
	_sizes.clear();
	_positions.clear();
	_colors.clear();
}



void GLBoxObject::update()
{
	for ( int i = 0; i < _nodes.size(); i++ )
	{
		const node_ref_t& ref = _nodes[i];
		const vec3_t& center = _db->graphs()[ref.graph_id]->positions()[ref.node_id];

		update_cube(i, center, _sizes[i]);
	}
}



void GLBoxObject::update_cube(int i, const vec3_t& C, float L)
{
	int j = i * VERTICES_PER_BOX;

	_positions.replace(j++, vec3_t { C.x - L/2, C.y + L/2, C.z - L/2 });
	_positions.replace(j++, vec3_t { C.x + L/2, C.y + L/2, C.z - L/2 });
	_positions.replace(j++, vec3_t { C.x + L/2, C.y + L/2, C.z - L/2 });
	_positions.replace(j++, vec3_t { C.x + L/2, C.y - L/2, C.z - L/2 });
	_positions.replace(j++, vec3_t { C.x + L/2, C.y - L/2, C.z - L/2 });
	_positions.replace(j++, vec3_t { C.x - L/2, C.y - L/2, C.z - L/2 });
	_positions.replace(j++, vec3_t { C.x - L/2, C.y - L/2, C.z - L/2 });
	_positions.replace(j++, vec3_t { C.x - L/2, C.y + L/2, C.z - L/2 });

	_positions.replace(j++, vec3_t { C.x - L/2, C.y + L/2, C.z + L/2 });
	_positions.replace(j++, vec3_t { C.x + L/2, C.y + L/2, C.z + L/2 });
	_positions.replace(j++, vec3_t { C.x + L/2, C.y + L/2, C.z + L/2 });
	_positions.replace(j++, vec3_t { C.x + L/2, C.y - L/2, C.z + L/2 });
	_positions.replace(j++, vec3_t { C.x + L/2, C.y - L/2, C.z + L/2 });
	_positions.replace(j++, vec3_t { C.x - L/2, C.y - L/2, C.z + L/2 });
	_positions.replace(j++, vec3_t { C.x - L/2, C.y - L/2, C.z + L/2 });
	_positions.replace(j++, vec3_t { C.x - L/2, C.y + L/2, C.z + L/2 });

	_positions.replace(j++, vec3_t { C.x - L/2, C.y + L/2, C.z - L/2 });
	_positions.replace(j++, vec3_t { C.x - L/2, C.y + L/2, C.z + L/2 });
	_positions.replace(j++, vec3_t { C.x + L/2, C.y + L/2, C.z - L/2 });
	_positions.replace(j++, vec3_t { C.x + L/2, C.y + L/2, C.z + L/2 });
	_positions.replace(j++, vec3_t { C.x + L/2, C.y - L/2, C.z - L/2 });
	_positions.replace(j++, vec3_t { C.x + L/2, C.y - L/2, C.z + L/2 });
	_positions.replace(j++, vec3_t { C.x - L/2, C.y - L/2, C.z - L/2 });
	_positions.replace(j++, vec3_t { C.x - L/2, C.y - L/2, C.z + L/2 });
}

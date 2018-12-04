#include <QKeyEvent>
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <QTimer>
#include <QWheelEvent>
#include "arguments.h"
#include "fdl.h"
#include "glwidget.h"



const int MAX_FPS = 60;



static const char *VERTEX_SHADER_SOURCE =
	"#version 150\n"
	"in vec4 position;\n"
	"in vec4 color;\n"
	"uniform mat4 mvpMatrix;\n"
	"out vec4 frag_color;\n"
	"void main() {\n"
	"   gl_Position = mvpMatrix * position;\n"
	"   gl_PointSize = 3.0;\n"
	"   frag_color = color;\n"
	"}\n";



static const char *FRAGMENT_SHADER_SOURCE =
	"#version 150\n"
	"in highp vec4 frag_color;\n"
	"out highp vec4 out_color;\n"
	"void main() {\n"
	"   out_color = frag_color;\n"
	"}\n";



/**
 * Construct a GL widget.
 *
 * @param db
 * @param parent
 */
GLWidget::GLWidget(Database *db, QWidget *parent):
	QOpenGLWidget(parent),
	_db(db)
{
	// get command-line arguments
	Arguments& args {Arguments::instance()};
	_fdl_3d = args.fdl_3d;

	// allow the widget to enter focus by being clicked
	setFocusPolicy(Qt::ClickFocus);
	
	// register the animation callback
	startTimer(1000 / MAX_FPS);
}



/**
 * Destruct a GL widget.
 */
GLWidget::~GLWidget()
{
	makeCurrent();

	for ( GLGraphObject *obj : _graphs )
	{
		delete obj;
	}

	for ( GLAlignObject *obj : _alignments )
	{
		delete obj;
	}

	delete _boxes;
	delete _program;

	doneCurrent();
}



/**
 * Rotate the model.
 *
 * @param deltaX
 * @param deltaY
 * @param deltaZ
 */
void GLWidget::rotate(float deltaX, float deltaY, float deltaZ)
{
	_rot.setX(_rot.x() + deltaX);
	_rot.setY(_rot.y() + deltaY);
	_rot.setZ(_rot.z() + deltaZ);

	_model.setToIdentity();
	_model.rotate(_rot.x(), 1, 0, 0);
	_model.rotate(_rot.y(), 0, 1, 0);
	_model.rotate(_rot.z(), 0, 0, 1);
}



/**
 * Set the list of selected nodes and render a box for each node.
 *
 * @param node
 */
void GLWidget::setSelectedNodes(const QVector<NodeRef>& nodes)
{
	_selected_nodes = nodes;

	_boxes->clear();
	_boxes->append(_selected_nodes, 3.0f, Color { 1, 0, 0, 1 });
	_boxes->update();
	update();
}



/**
 * Translate the view.
 *
 * @param deltaX
 * @param deltaY
 * @param deltaZ
 */
void GLWidget::translate(float deltaX, float deltaY, float deltaZ)
{
	_view.translate(deltaX, deltaY, deltaZ);
}



/**
 * Change the zoom level.
 *
 * @param delta
 */
void GLWidget::zoom(float delta)
{
	_zoom = qMin(qMax(1.0f, _zoom + delta), 180.0f);

	_proj.setToIdentity();
	_proj.perspective(
		_zoom,
		float(width()) / height(),
		0.0001f, 1000.0f
	);
}



/**
 * Initialize the model, view, and projection matrices.
 */
void GLWidget::initializeCamera()
{
	// initialize model matrix
	_rot = QVector3D(0, 0, 0);
	rotate(0, 0, 0);

	// initialize view matrix (camera)
	_view.setToIdentity();
	translate(0, 0, -400);

	// initialize projection matrix
	_zoom = 60.0f;
	zoom(0);
}



/**
 * Perform a single time-step of the animation.
 */
void GLWidget::runAnimation()
{
	static bool running = false;

	// skip this frame if previous frame is still processing
	if ( running )
	{
		return;
	}

	running = true;

	// perform FDL on GPU if it is enabled
	if ( _fdl_gpu )
	{
		for ( Graph *g : _db->graphs().values() )
		{
			// execute FDL kernel on GPU
			if ( _fdl_3d )
			{
				fdl_3d_gpu(
					g->nodes().size(),
					g->positions_gpu(),
					g->velocities_gpu(),
					g->edge_matrix_gpu()
				);
			}
			else
			{
				fdl_2d_gpu(
					g->nodes().size(),
					g->positions_gpu(),
					g->velocities_gpu(),
					g->edge_matrix_gpu()
				);
			}

			// read position data from GPU
			g->gpu_read_positions();
		}

		// wait for GPU to process all graphs
		CUDA_SAFE_CALL(cudaStreamSynchronize(0));
	}

	// otherwise perform FDL on CPU
	else
	{
		for ( Graph *g : _db->graphs().values() )
		{
			if ( _fdl_3d )
			{
				fdl_3d_cpu(
					g->nodes().size(),
					g->positions().data(),
					g->velocities().data(),
					g->edge_matrix().data()
				);
			}
			else
			{
				fdl_2d_cpu(
					g->nodes().size(),
					g->positions().data(),
					g->velocities().data(),
					g->edge_matrix().data()
				);
			}
		}
	}

	// update alignment positions
	for ( Alignment *a : _db->alignments() )
	{
		a->update();
	}

	// update box positions
	_boxes->update();

	// render this frame
	update();

	running = false;
}



/**
 * Initialize OpenGL resources.
 */
void GLWidget::initializeGL()
{
	initializeOpenGLFunctions();
	glClearColor(1, 1, 1, 0);
	glDepthMask(GL_FALSE);
	glEnable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);

	_program = new QOpenGLShaderProgram;
	_program->addShaderFromSourceCode(QOpenGLShader::Vertex, VERTEX_SHADER_SOURCE);
	_program->addShaderFromSourceCode(QOpenGLShader::Fragment, FRAGMENT_SHADER_SOURCE);
	_program->bindAttributeLocation("position", 0);
	_program->bindAttributeLocation("color", 1);
	_program->link();

	_program->bind();
	_ref_mvp_matrix = _program->uniformLocation("mvpMatrix");

	// initialize scene object for each graph
	for ( Graph *g : _db->graphs().values() )
	{
		GLGraphObject *obj = new GLGraphObject(g);
		obj->initialize();

		_graphs.push_back(obj);
	}

	// initialize scene object for each alignment
	for ( Alignment *a : _db->alignments() )
	{
		GLAlignObject *obj = new GLAlignObject(a);
		obj->initialize();

		_alignments.push_back(obj);
	}

	// initialize scene object for boxes
	_boxes = new GLBoxObject(_db);
	_boxes->initialize();

	// initialize camera
	initializeCamera();

	_program->release();
}



/**
 * Render the current frame.
 */
void GLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set MVP matrix in shader program
	_program->bind();
	_program->setUniformValue(_ref_mvp_matrix, _proj * _view * _model);

	// draw each alignment
	if ( _show_alignments )
	{
		for ( GLAlignObject *obj : _alignments )
		{
			obj->paint();
		}
	}

	// draw each graph
	if ( _show_graphs )
	{
		for ( GLGraphObject *obj : _graphs )
		{
			obj->paint(_show_modules);
		}
	}

	// draw boxes for selected nodes
	_boxes->paint();

	_program->release();
}



/**
 * Resize the GL widget.
 */
void GLWidget::resizeGL(int, int)
{
	zoom(0);
}



/**
 * Handle key-press events.
 *
 * @param event
 */
void GLWidget::keyPressEvent(QKeyEvent *event)
{
	const float ROT_DELTA = 1;
	const float TRANS_DELTA = 10;
	const float ZOOM_DELTA = 1;

	switch ( event->key() )
	{
	case Qt::Key_R:
		initializeCamera();
		break;
	case Qt::Key_W:
		translate(0, -TRANS_DELTA, 0);
		break;
	case Qt::Key_S:
		translate(0, +TRANS_DELTA, 0);
		break;
	case Qt::Key_A:
		translate(+TRANS_DELTA, 0, 0);
		break;
	case Qt::Key_D:
		translate(-TRANS_DELTA, 0, 0);
		break;
	case Qt::Key_Q:
		zoom(+ZOOM_DELTA);
		break;
	case Qt::Key_E:
		zoom(-ZOOM_DELTA);
		break;
	case Qt::Key_I:
		rotate(+ROT_DELTA, 0, 0);
		break;
	case Qt::Key_K:
		rotate(-ROT_DELTA, 0, 0);
		break;
	case Qt::Key_J:
		rotate(0, +ROT_DELTA, 0);
		break;
	case Qt::Key_L:
		rotate(0, -ROT_DELTA, 0);
		break;
	case Qt::Key_U:
		rotate(0, 0, +ROT_DELTA);
		break;
	case Qt::Key_O:
		rotate(0, 0, -ROT_DELTA);
	case Qt::Key_G:
		_fdl_gpu = !_fdl_gpu;

		// copy graph data to GPU if it is enabled
		if ( _fdl_gpu )
		{
			for ( Graph *g : _db->graphs().values() )
			{
				g->gpu_write_positions();
				g->gpu_write_velocities();
			}

			CUDA_SAFE_CALL(cudaStreamSynchronize(0));
		}

		// otherwise copy graph data to CPU
		else
		{
			for ( Graph *g : _db->graphs().values() )
			{
				g->gpu_read_positions();
				g->gpu_read_velocities();
			}
		}
		break;
	case Qt::Key_Space:
		_animate = !_animate;
		break;
	case Qt::Key_Z:
		_show_graphs = !_show_graphs;
		break;
	case Qt::Key_X:
		_show_alignments = !_show_alignments;
		break;
	case Qt::Key_C:
		_show_modules = !_show_modules;
		break;
	case Qt::Key_B:
		_select_multi = !_select_multi;
		break;
	}

	update();
	event->accept();
}



/**
 * Handle double-click events. Double-clicking selects a node in a graph, or a
 * group of nodes if multi-select is enabled.
 *
 * @param event
 */
void GLWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
	float max_dist = _select_multi ? 5.0f : 40.0f;
	float x = event->x();
	float y = height() - event->y();
	QMatrix4x4 mv = _view * _model;
	QVector3D start = QVector3D(x, y, 0.0f).unproject(mv, _proj, rect());
	QVector3D end = QVector3D(x, y, 1.0f).unproject(mv, _proj, rect());
	QVector3D dir = (end - start).normalized();

	QVector<NodeRef> nodes;
	float min_dist = max_dist;

	for ( int i = 0; i < _db->graphs().size(); i++ )
	{
		Graph *g = _db->graphs().values()[i];

		for ( int j = 0; j < g->nodes().size(); j++ )
		{
			Vector3 v = g->positions()[j];
			float dist = QVector3D(v.x, v.y, v.z).distanceToLine(start, dir);

			if ( dist < max_dist )
			{
				if ( _select_multi )
				{
					nodes.push_back(NodeRef { g->id(), j });
				}
				else if ( dist < min_dist )
				{
					min_dist = dist;

					nodes.clear();
					nodes.push_back(NodeRef { g->id(), j });
				}
			}
		}
	}

	emit nodesSelected(nodes);
	event->accept();
}



/**
 * Handle mouse-move events. Dragging the mouse rotates the model.
 *
 * @param event
 */
void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
	int dx = event->x() - _prev_pos.x();
	int dy = event->y() - _prev_pos.y();

	if ( event->buttons() & Qt::LeftButton )
	{
		rotate(0.5f * dy, 0.5f * dx, 0);
	}
	else if ( event->buttons() & Qt::RightButton )
	{
		rotate(0.5f * dy, 0, 0.5f * dx);
	}
	_prev_pos = event->pos();

	update();
	event->accept();
}



/**
 * Handle mouse-move events. Clicking the mouse sets the "previous position"
 * which is used to rotate the model when dragging the mouse.
 *
 * @param event
 */
void GLWidget::mousePressEvent(QMouseEvent *event)
{
	_prev_pos = event->pos();
	event->accept();
}



/**
 * Handle timer events. The timer is used to run the animation.
 *
 * @param event
 */
void GLWidget::timerEvent(QTimerEvent *event)
{
	if ( _animate )
	{
		runAnimation();
	}

	event->accept();
}



/**
 * Handle mouse-wheel events. Scrolling is used to zoom in and out.
 *
 * @param event
 */
void GLWidget::wheelEvent(QWheelEvent *event)
{
	QPoint pixels = event->pixelDelta();
	QPoint degrees = event->angleDelta();

	if ( !pixels.isNull() )
	{
		zoom(-pixels.y() / 10.0f);
	}
	else if ( !degrees.isNull() )
	{
		zoom(-degrees.y() / 10.0f);
	}

	update();
	event->accept();
}

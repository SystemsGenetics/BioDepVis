#include <QKeyEvent>
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <QTimer>
#include <QWheelEvent>
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

GLWidget::GLWidget(Database *db, QWidget *parent)
    : QOpenGLWidget(parent),
      _db(db),
      _animate(false),
      _gpu(true),
      _select_multi(false),
      _show_alignments(true),
      _show_graphs(true),
      _show_modules(false),
      _rot(0, 0, 0),
      _zoom(0),
      _program(0)
{
    setFocusPolicy(Qt::ClickFocus);

    startTimer(1000 / MAX_FPS);
}

GLWidget::~GLWidget()
{
    makeCurrent();

    for ( GLGraphObject *obj : _graphs ) {
        delete obj;
    }

    for ( GLAlignObject *obj : _alignments ) {
        delete obj;
    }

    delete _boxes;
    delete _program;

    doneCurrent();
}

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

void GLWidget::setSelectedNodes(const QVector<node_ref_t>& nodes)
{
    _selected_nodes = nodes;

    _boxes->clear();
    _boxes->append(_selected_nodes, 3.0f, color_t { 1, 0, 0, 1 });
    _boxes->update();
    update();
}

void GLWidget::translate(float deltaX, float deltaY, float deltaZ)
{
    _view.translate(deltaX, deltaY, deltaZ);
}

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

void GLWidget::init_camera()
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

void GLWidget::run_animation()
{
    static bool running = false;

    if ( running ) {
        return;
    }

    running = true;

    for ( Graph *g : _db->graphs().values() ) {
        if ( _gpu ) {
            fdl_2d_gpu(
                g->nodes().size(),
                g->positions_gpu(),
                g->positions_d_gpu(),
                g->edge_matrix_gpu()
            );
        }
        else {
            fdl_2d_cpu(
                g->nodes().size(),
                g->positions().data(),
                g->positions_d().data(),
                g->edge_matrix().data()
            );
        }
    }

    if ( _gpu ) {
        gpu_sync();

        // copy graph data from GPU
        for ( Graph *g : _db->graphs().values() ) {
            g->read_gpu();
        }
    }

    for ( Alignment *a : _db->alignments() ) {
        a->update();
    }

    _boxes->update();
    update();

    running = false;
}

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
    for ( Graph *g : _db->graphs().values() ) {
        GLGraphObject *obj = new GLGraphObject(g);
        obj->initialize();

        _graphs.push_back(obj);
    }

    // initialize scene object for each alignment
    for ( Alignment *a : _db->alignments() ) {
        GLAlignObject *obj = new GLAlignObject(a);
        obj->initialize();

        _alignments.push_back(obj);
    }

    // initialize scene object for boxes
    _boxes = new GLBoxObject(_db);
    _boxes->initialize();

    // initialize camera
    init_camera();

    _program->release();
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set MVP matrix in shader program
    _program->bind();
    _program->setUniformValue(_ref_mvp_matrix, _proj * _view * _model);

    // draw each alignment
    if ( _show_alignments ) {
        for ( GLAlignObject *obj : _alignments ) {
            obj->paint();
        }
    }

    // draw each graph
    if ( _show_graphs ) {
        for ( GLGraphObject *obj : _graphs ) {
            obj->paint(_show_modules);
        }
    }

    // draw boxes for selected nodes
    _boxes->paint();

    _program->release();
}

void GLWidget::resizeGL(int /*w*/, int /*h*/)
{
    zoom(0);
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
    const float ROT_DELTA = 1;
    const float TRANS_DELTA = 10;
    const float ZOOM_DELTA = 1;

    switch ( event->key() ) {
    case Qt::Key_R:
        init_camera();
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
        _gpu = !_gpu;

        if ( _gpu ) {
            // copy graph data to GPU
            for ( Graph *g : _db->graphs().values() ) {
                g->write_gpu();
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

void GLWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    float max_dist = _select_multi ? 5.0f : 40.0f;
    float x = event->x();
    float y = height() - event->y();
    QMatrix4x4 mv = _view * _model;
    QVector3D start = QVector3D(x, y, 0.0f).unproject(mv, _proj, rect());
    QVector3D end = QVector3D(x, y, 1.0f).unproject(mv, _proj, rect());
    QVector3D dir = (end - start).normalized();

    QVector<node_ref_t> nodes;
    float min_dist = max_dist;

    for ( int i = 0; i < _db->graphs().size(); i++ ) {
        Graph *g = _db->graphs().values()[i];

        for ( int j = 0; j < g->nodes().size(); j++ ) {
            vec3_t v = g->positions()[j];
            float dist = QVector3D(v.x, v.y, v.z).distanceToLine(start, dir);

            if ( dist < max_dist ) {
                if ( _select_multi ) {
                    nodes.push_back(node_ref_t { g->id(), j });
                }
                else if ( dist < min_dist ) {
                    min_dist = dist;

                    nodes.clear();
                    nodes.push_back(node_ref_t { g->id(), j });
                }
            }
        }
    }

    emit nodesSelected(nodes);
    event->accept();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - _prev_pos.x();
    int dy = event->y() - _prev_pos.y();

    if ( event->buttons() & Qt::LeftButton ) {
        rotate(0.5f * dy, 0.5f * dx, 0);
    }
    else if ( event->buttons() & Qt::RightButton ) {
        rotate(0.5f * dy, 0, 0.5f * dx);
    }
    _prev_pos = event->pos();

    update();
    event->accept();
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    _prev_pos = event->pos();
    event->accept();
}

void GLWidget::timerEvent(QTimerEvent *event)
{
    if ( _animate ) {
        run_animation();
    }

    event->accept();
}

void GLWidget::wheelEvent(QWheelEvent *event)
{
    QPoint pixels = event->pixelDelta();
    QPoint degrees = event->angleDelta();

    if ( !pixels.isNull() ) {
        zoom(-pixels.y() / 10.0f);
    }
    else if ( !degrees.isNull() ) {
        zoom(-degrees.y() / 10.0f);
    }

    update();
    event->accept();
}

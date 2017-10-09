#include <QKeyEvent>
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <QTimer>
#include <QWheelEvent>
#include "fdl.h"
#include "glwidget.h"

const int MAX_FPS = 30;

const color_t GRAPH_EDGE_COLORS[] = {
    { 0.65f, 0.81f, 0.89f, 0.30f },
    { 0.50f, 0.50f, 0.78f, 0.30f },
    { 0.42f, 0.24f, 0.60f, 0.30f },
    { 0.12f, 0.47f, 0.71f, 0.30f },
    { 0.70f, 0.87f, 0.54f, 0.30f },
    { 1.00f, 0.50f, 0.00f, 0.30f }
};

const color_t ALIGN_EDGE_COLORS[] = {
    { 0.70f, 0.19f, 0.29f, 0.10f },
    { 0.89f, 0.50f, 0.79f, 0.10f },
    { 0.89f, 0.50f, 0.79f, 0.10f }
};

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
      _alignment(true),
      _animate(false),
      _gpu(true),
      _module_color(false),
      _select_multi(false),
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

    for ( GraphObject *obj : _graphs ) {
        obj->vbo_coords.destroy();
        obj->vbo_colors.destroy();
        delete obj;
    }

    for ( AlignObject *obj : _alignments ) {
        obj->vbo_coords.destroy();
        obj->vbo_colors.destroy();
        delete obj;
    }

    delete _program;
    _program = 0;

    doneCurrent();
}

void GLWidget::setRotX(float angle)
{
    _rot.setX(angle);
}

void GLWidget::setRotY(float angle)
{
    _rot.setY(angle);
}

void GLWidget::setRotZ(float angle)
{
    _rot.setZ(angle);
}

void GLWidget::setSelectedNodes(const QVector<node_ref_t>& nodes)
{
    _selected_nodes = nodes;
}

void GLWidget::setZoom(float zoom)
{
    _zoom = qMin(qMax(1.0f, zoom), 180.0f);
}

void GLWidget::init_camera()
{
    // initialize rotation angle
    _rot.setX(0);
    _rot.setY(0);
    _rot.setZ(0);

    // initialize zoom
    _zoom = 60.0f;

    // initialize view matrix (camera)
    _view.setToIdentity();
    _view.translate(0, 0, -400);
}

void GLWidget::run_animation()
{
    for ( Graph *g : _db->graphs().values() ) {
        if ( _gpu ) {
            fdl_2d_gpu(
                g->nodes().size(),
                g->coords_gpu(),
                g->coords_d_gpu(),
                g->edge_matrix_gpu()
            );
        }
        else {
            fdl_2d_cpu(
                g->nodes().size(),
                g->coords().data(),
                g->coords_d().data(),
                g->edge_matrix().data()
            );
        }
    }

    if ( _gpu ) {
        gpu_sync();

        // copy graph data from GPU
        for ( Graph *g : _db->graphs().values() ) {
            int n = g->nodes().size();
            gpu_read(g->coords().data(), g->coords_gpu(), n * sizeof(vec3_t));
            gpu_read(g->coords_d().data(), g->coords_d_gpu(), n * sizeof(vec3_t));
            gpu_read(g->edge_matrix().data(), g->edge_matrix_gpu(), n * n * sizeof(bool));
        }
    }

    for ( Alignment *a : _db->alignments() ) {
        a->update();
    }

    update();
}

void GLWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glClearColor(1, 1, 1, 0);
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
    for ( int i = 0; i < _db->graphs().size(); i++ ) {
        Graph *g = _db->graphs().values()[i];
        GraphObject *obj = new GraphObject();
        obj->g = g;

        // initialize node colors
        obj->node_colors.reserve(g->nodes().size());
        for ( int j = 0; j < g->nodes().size(); j++ ) {
            obj->node_colors.push_back(color_t { 0, 0, 0, 1 });
        }

        // initialize edge colors
        obj->edge_colors.reserve(g->edges().size());
        for ( int j = 0; j < g->edges().size(); j++ ) {
            obj->edge_colors.push_back(GRAPH_EDGE_COLORS[i]);
        }

        // initialize vertex array object
        obj->vao.create();

        QOpenGLVertexArrayObject::Binder vaoBinder(&obj->vao);

        // initialize position buffer
        obj->vbo_coords.create();
        obj->vbo_coords.bind();
        obj->vbo_coords.allocate(g->nodes().size() * sizeof(vec3_t));

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        obj->vbo_coords.release();

        // initialize color buffer
        int num_colors = qMax(g->nodes().size(), g->edges().size());

        obj->vbo_colors.create();
        obj->vbo_colors.bind();
        obj->vbo_colors.allocate(num_colors * sizeof(color_t));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
        obj->vbo_colors.release();

        _graphs.push_back(obj);
    }

    // initialize scene object for each alignment
    for ( int i = 0; i < _db->alignments().size(); i++ ) {
        Alignment *a = _db->alignments()[i];
        AlignObject *obj = new AlignObject();
        obj->a = a;

        // initialize edge colors
        obj->edge_colors.reserve(a->edges().size());
        for ( int j = 0; j < a->edges().size(); j++ ) {
            obj->edge_colors.push_back(ALIGN_EDGE_COLORS[i]);
        }

        // initialize vertex array object
        obj->vao.create();

        QOpenGLVertexArrayObject::Binder vaoBinder(&obj->vao);

        // initialize position buffer
        obj->vbo_coords.create();
        obj->vbo_coords.bind();
        obj->vbo_coords.allocate(a->edges().size() * sizeof(align_edge_t));

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        obj->vbo_coords.release();

        // initialize color buffer
        obj->vbo_colors.create();
        obj->vbo_colors.bind();
        obj->vbo_colors.allocate(a->edges().size() * sizeof(color_t));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
        obj->vbo_colors.release();

        _alignments.push_back(obj);
    }

    // initialize camera
    init_camera();

    _program->release();
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

    // compute model matrix
    _model.setToIdentity();
    _model.rotate(_rot.x(), 1, 0, 0);
    _model.rotate(_rot.y(), 0, 1, 0);
    _model.rotate(_rot.z(), 0, 0, 1);

    // compute projection matrix
    _proj.setToIdentity();
    _proj.perspective(
        _zoom,
        float(width()) / height(),
        0.0001f, 1000.0f
    );

    // set MVP matrix in shader program
    _program->bind();
    _program->setUniformValue(_ref_mvp_matrix, _proj * _view * _model);

    // draw each graph
    for ( GraphObject *obj : _graphs ) {
        QOpenGLVertexArrayObject::Binder vaoBinder(&obj->vao);

        // write node positions
        obj->vbo_coords.bind();
        obj->vbo_coords.write(0, obj->g->coords().data(), obj->g->nodes().size() * sizeof(vec3_t));
        obj->vbo_coords.release();

        // write node colors
        const color_t *color_data = _module_color
            ? obj->g->colors().data()
            : obj->node_colors.data();

        obj->vbo_colors.bind();
        obj->vbo_colors.write(0, color_data, obj->g->nodes().size() * sizeof(color_t));
        obj->vbo_colors.release();

        // draw nodes
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_POINTS, 0, obj->g->coords().size());

        // write edge colors
        obj->vbo_colors.bind();
        obj->vbo_colors.write(0, obj->edge_colors.data(), obj->g->edges().size() * sizeof(color_t));
        obj->vbo_colors.release();

        // draw edges
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glLineWidth(0.001f);
        glDrawElements(GL_LINES, obj->g->edges().size(), GL_UNSIGNED_INT, obj->g->edges().data());
    }

    // draw each alignment
    if ( _alignment ) {
        for ( AlignObject *obj : _alignments ) {
            QOpenGLVertexArrayObject::Binder vaoBinder(&obj->vao);

            // write edge positions
            obj->vbo_coords.bind();
            obj->vbo_coords.write(0, obj->a->vertices().data(), obj->a->edges().size() * sizeof(align_edge_t));
            obj->vbo_coords.release();

            // write edge colors
            obj->vbo_colors.bind();
            obj->vbo_colors.write(0, obj->edge_colors.data(), obj->a->edges().size() * sizeof(color_t));
            obj->vbo_colors.release();

            // draw edges
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glLineWidth(0.1f);
            glDrawArrays(GL_LINES, 0, obj->a->edges().size());
        }
    }

    _program->release();
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
    const float SHIFT_ROT = 1;
    const float SHIFT_TRANS = 10;
    const float SHIFT_ZOOM = 1;

    switch ( event->key() ) {
    case Qt::Key_R:
        init_camera();
        break;
    case Qt::Key_W:
        _view.translate(0, -SHIFT_TRANS, 0);
        break;
    case Qt::Key_S:
        _view.translate(0, +SHIFT_TRANS, 0);
        break;
    case Qt::Key_A:
        _view.translate(+SHIFT_TRANS, 0, 0);
        break;
    case Qt::Key_D:
        _view.translate(-SHIFT_TRANS, 0, 0);
        break;
    case Qt::Key_Q:
        setZoom(_zoom + SHIFT_ZOOM);
        break;
    case Qt::Key_E:
        setZoom(_zoom - SHIFT_ZOOM);
        break;
    case Qt::Key_I:
        setRotX(_rot.x() + SHIFT_ROT);
        break;
    case Qt::Key_K:
        setRotX(_rot.x() - SHIFT_ROT);
        break;
    case Qt::Key_J:
        setRotY(_rot.y() + SHIFT_ROT);
        break;
    case Qt::Key_L:
        setRotY(_rot.y() - SHIFT_ROT);
        break;
    case Qt::Key_U:
        setRotZ(_rot.z() + SHIFT_ROT);
        break;
    case Qt::Key_O:
        setRotZ(_rot.z() - SHIFT_ROT);
    case Qt::Key_G:
        _gpu = !_gpu;
        break;
    case Qt::Key_Space:
        _animate = !_animate;
        break;
    case Qt::Key_C:
        _module_color = !_module_color;
        break;
    case Qt::Key_V:
        _alignment = !_alignment;
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
    const float max_dist = 5.0f;
    float x = event->x();
    float y = height() - event->y();
    QMatrix4x4 mv = _model * _view;
    QVector3D start = QVector3D(x, y, 0.0f).unproject(mv, _proj, rect());
    QVector3D end = QVector3D(x, y, 1.0f).unproject(mv, _proj, rect());
    QVector3D dir = (end - start).normalized();

    QVector<node_ref_t> nodes;
    float min_dist = max_dist;

    for ( int i = 0; i < _db->graphs().size(); i++ ) {
        Graph *g = _db->graphs().values()[i];

        for ( int j = 0; j < g->nodes().size(); j++ ) {
            vec3_t v = g->coords()[j];
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
        setRotX(_rot.x() + 0.5f * dy);
        setRotY(_rot.y() + 0.5f * dx);
    }
    else if ( event->buttons() & Qt::RightButton ) {
        setRotX(_rot.x() + 0.5f * dy);
        setRotZ(_rot.z() + 0.5f * dx);
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
        setZoom(_zoom - pixels.y() / 10.0f);
    }
    else if ( !degrees.isNull() ) {
        setZoom(_zoom - degrees.y() / 10.0f);
    }

    update();
    event->accept();
}

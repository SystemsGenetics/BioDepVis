#include <QKeyEvent>
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include "glwidget.h"

static const char *VERTEX_SHADER_SOURCE =
    "#version 150\n"
    "in vec4 position;\n"
    "uniform mat4 mvpMatrix;\n"
    "void main() {\n"
    "   gl_Position = mvpMatrix * position;\n"
    "   gl_PointSize = 5.0;\n"
    "}\n";

static const char *FRAGMENT_SHADER_SOURCE =
    "#version 150\n"
    "out highp vec4 out_color;\n"
    "void main() {\n"
    "   out_color = vec4(0.0, 0.0, 0.0, 1.0);\n"
    "}\n";

GLWidget::GLWidget(Database *db, QWidget *parent)
    : QOpenGLWidget(parent),
      _rot(0, 0, 0),
      _zoom(0),
      _db(db),
      _program(0)
{
    setFocusPolicy(Qt::ClickFocus);
}

GLWidget::~GLWidget()
{
    makeCurrent();

    for ( GraphObject *obj : _graphs ) {
        obj->vbo.destroy();
        delete obj;
    }

    delete _program;
    _program = 0;

    doneCurrent();
}

static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}

void GLWidget::setRotX(int angle)
{
    qNormalizeAngle(angle);
    _rot.setX(angle);
}

void GLWidget::setRotY(int angle)
{
    qNormalizeAngle(angle);
    _rot.setY(angle);
}

void GLWidget::setRotZ(int angle)
{
    qNormalizeAngle(angle);
    _rot.setZ(angle);
}

void GLWidget::init_camera()
{
    // initialize rotation angle
    _rot.setX(0);
    _rot.setY(0);
    _rot.setZ(0);

    // initialize zoom
    _zoom = 0;

    // initialize view matrix (camera)
    _view.setToIdentity();
    _view.translate(0, 0, -400);
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
    _program->link();

    _program->bind();
    _ref_mvp_matrix = _program->uniformLocation("mvpMatrix");

    // initialize scene object for each graph
    for ( Graph *g : _db->graphs().values() ) {
        GraphObject *obj = new GraphObject();
        obj->g = g;

        // initialize vertex array object
        obj->vao.create();

        QOpenGLVertexArrayObject::Binder vaoBinder(&obj->vao);

        // initialize buffer for node positions
        obj->vbo.create();
        obj->vbo.bind();
        obj->vbo.allocate(g->coords().data(), g->coords().size() * sizeof(vec3_t));

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        obj->vbo.release();

        _graphs.push_back(obj);
    }

    // initialize camera
    init_camera();

    _program->release();
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // compute model matrix
    _model.setToIdentity();
    _model.rotate(_rot.x() / 16.0f, 1, 0, 0);
    _model.rotate(_rot.y() / 16.0f, 0, 1, 0);
    _model.rotate(_rot.z() / 16.0f, 0, 0, 1);

    // compute projection matrix
    _proj.setToIdentity();
    _proj.perspective(
        qMin(qMax(1.0f, 45.0f + _zoom), 270.0f),
        float(width()) / height(),
        0.0001f, 1000.0f
    );

    // draw each graph
    for ( GraphObject *obj : _graphs ) {
        QOpenGLVertexArrayObject::Binder vaoBinder(&obj->vao);

        // set MVP matrix in shader program
        _program->bind();
        _program->setUniformValue(_ref_mvp_matrix, _proj * _view * _model);

        // draw nodes
        glDrawArrays(GL_POINTS, 0, obj->g->coords().size());

        // TODO: draw edges
        // glLineWidth(0.001f);
        // glDrawElements(GL_LINES, obj->g->edges().size() * sizeof(graph_edge_t), GL_UNSIGNED_INT, obj->g->edges().data());
    }

    _program->release();
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
    const float SHIFT_ROT = 16;
    const float SHIFT_TRANS = 10;

    switch ( event->key() ) {
    case Qt::Key_R:
        init_camera();
        break;
    case Qt::Key_I:
        _view.translate(0, -SHIFT_TRANS, 0);
        break;
    case Qt::Key_K:
        _view.translate(0, +SHIFT_TRANS, 0);
        break;
    case Qt::Key_J:
        _view.translate(+SHIFT_TRANS, 0, 0);
        break;
    case Qt::Key_L:
        _view.translate(-SHIFT_TRANS, 0, 0);
        break;
    case Qt::Key_W:
        setRotX(_rot.x() + SHIFT_ROT);
        break;
    case Qt::Key_S:
        setRotX(_rot.x() - SHIFT_ROT);
        break;
    case Qt::Key_A:
        setRotY(_rot.y() + SHIFT_ROT);
        break;
    case Qt::Key_D:
        setRotY(_rot.y() - SHIFT_ROT);
        break;
    case Qt::Key_Q:
        _zoom += 1.0f;
        break;
    case Qt::Key_E:
        _zoom -= 1.0f;
        break;
    }

    update();
    QOpenGLWidget::keyPressEvent(event);
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    _prev_pos = event->pos();
    QOpenGLWidget::mousePressEvent(event);
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - _prev_pos.x();
    int dy = event->y() - _prev_pos.y();

    if ( event->buttons() & Qt::LeftButton ) {
        setRotX(_rot.x() + 8 * dy);
        setRotY(_rot.y() + 8 * dx);
        update();
    }
    else if ( event->buttons() & Qt::RightButton ) {
        setRotX(_rot.x() + 8 * dy);
        setRotZ(_rot.z() + 8 * dx);
        update();
    }
    _prev_pos = event->pos();
    QOpenGLWidget::mousePressEvent(event);
}

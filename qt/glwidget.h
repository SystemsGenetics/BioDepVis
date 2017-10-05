#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QOpenGLVertexArrayObject>
#include "database.h"

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)

typedef struct {
    Graph *g;
    QOpenGLVertexArrayObject vao;
    QOpenGLBuffer vbo;
} GraphObject;

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    GLWidget(Database *db, QWidget *parent = Q_NULLPTR);
    ~GLWidget();

public slots:
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int width, int height) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;

private:
    QVector3D _rot;
    QPoint _prev_pos;
    Database *_db;
    QVector<GraphObject *> _graphs;
    QOpenGLShaderProgram *_program;
    int _ref_mvp_matrix;
    int _ref_normal_matrix;
    int _ref_light_pos;
    QMatrix4x4 _model;
    QMatrix4x4 _view;
    QMatrix4x4 _proj;
};

#endif

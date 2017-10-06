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
    QColor edge_color;
    QOpenGLVertexArrayObject vao;
    QOpenGLBuffer vbo;
} GraphObject;

typedef struct {
    Alignment *a;
    QColor edge_color;
    QOpenGLVertexArrayObject vao;
    QOpenGLBuffer vbo;
} AlignObject;

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    GLWidget(Database *db, QWidget *parent = Q_NULLPTR);
    ~GLWidget();

public slots:
    void setRotX(float angle);
    void setRotY(float angle);
    void setRotZ(float angle);
    void setZoom(float zoom);

protected:
    void init_camera();
    void initializeGL() override;
    void paintGL() override;
    void keyPressEvent(QKeyEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    Database *_db;
    bool _show_alignment;

    QPoint _prev_pos;
    QVector3D _rot;
    float _zoom;

    QOpenGLShaderProgram *_program;
    int _ref_mvp_matrix;
    int _ref_color;
    QMatrix4x4 _model;
    QMatrix4x4 _view;
    QMatrix4x4 _proj;

    QVector<GraphObject *> _graphs;
    QVector<AlignObject *> _alignments;
};

#endif

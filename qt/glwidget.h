#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QOpenGLVertexArrayObject>
#include "database.h"
#include "glalignobject.h"
#include "glgraphobject.h"

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    GLWidget(Database *db, QWidget *parent = Q_NULLPTR);
    ~GLWidget();

public slots:
    void rotate(float deltaX, float deltaY, float deltaZ);
    void setSelectedNodes(const QVector<node_ref_t>& nodes);
    void translate(float deltaX, float deltaY, float deltaZ);
    void zoom(float delta);

signals:
    void nodesSelected(const QVector<node_ref_t>& nodes);

protected:
    void init_camera();
    void run_animation();
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void keyPressEvent(QKeyEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void timerEvent(QTimerEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    Database *_db;
    bool _animate;
    bool _gpu;
    bool _roi;
    bool _select_multi;
    bool _show_alignments;
    bool _show_graphs;
    bool _show_modules;

    QPoint _prev_pos;
    QVector3D _rot;
    float _zoom;
    QVector<node_ref_t> _selected_nodes;

    QOpenGLShaderProgram *_program;
    int _ref_mvp_matrix;
    QMatrix4x4 _model;
    QMatrix4x4 _view;
    QMatrix4x4 _proj;

    QVector<GLGraphObject *> _graphs;
    QVector<GLAlignObject *> _alignments;
};

#endif

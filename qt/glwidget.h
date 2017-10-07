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
    QVector<color_t> node_colors;
    QVector<color_t> edge_colors;
    QOpenGLVertexArrayObject vao;
    QOpenGLBuffer vbo_coords;
    QOpenGLBuffer vbo_colors;
} GraphObject;

typedef struct {
    Alignment *a;
    QVector<color_t> edge_colors;
    QOpenGLVertexArrayObject vao;
    QOpenGLBuffer vbo_coords;
    QOpenGLBuffer vbo_colors;
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
    void setSelectedNodes(const QVector<node_ref_t>& nodes);
    void setZoom(float zoom);

signals:
    void nodesSelected(const QVector<node_ref_t>& nodes);

protected:
    void init_camera();
    void initializeGL() override;
    void paintGL() override;
    void keyPressEvent(QKeyEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    Database *_db;
    bool _alignment;
    bool _module_color;
    bool _select_multi;

    QPoint _prev_pos;
    QVector3D _rot;
    float _zoom;
    QVector<node_ref_t> _selected_nodes;

    QOpenGLShaderProgram *_program;
    int _ref_mvp_matrix;
    QMatrix4x4 _model;
    QMatrix4x4 _view;
    QMatrix4x4 _proj;

    QVector<GraphObject *> _graphs;
    QVector<AlignObject *> _alignments;
};

#endif

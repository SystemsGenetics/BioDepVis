#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include "alignment.h"
#include "database.h"
#include "graph.h"

class VisualizerWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
private:
    GLuint _textures;

    Database *_db;
    bool _animate;
    bool _showalignment;

public:
    VisualizerWidget(Database *db, QWidget *parent=Q_NULLPTR);

private:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();

    void draw_graph(const Graph& g);
    void draw_alignment(Alignment& a);
};

#endif

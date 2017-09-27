#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include "graph.h"
#include "alignment.h"

class VisualizerWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
private:
    GLuint textures;

public:
    VisualizerWidget(QWidget *parent=Q_NULLPTR) : QOpenGLWidget(parent) {};

private:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void drawGraph(Graph *g);
    void drawAlignment(Alignment *align);


};
#endif

#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <QOpenGLFunctions>
#include <QOpenGLWidget>

class VisualizerWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    VisualizerWidget(QWidget *parent=Q_NULLPTR) : QOpenGLWidget(parent) {};

private:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
};
#endif

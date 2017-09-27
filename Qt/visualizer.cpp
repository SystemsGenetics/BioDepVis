#include "visualizer.h"

void VisualizerWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}

void VisualizerWidget::resizeGL(int w, int h)
{
}

void VisualizerWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);
}

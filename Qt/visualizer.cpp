#include "visualizer.h"

void VisualizerWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glShadeModel(GL_SMOOTH);
    glDepthRange(0.0, 1.0);

    // TODO: loadTexture()

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);

    // initialize lights
    float light_position1[] = { 2.0, -200.0, 0.0, 0.0 };
    float light_position2[] = { 0.0, 10, -5.0, 0.0 };
    float light_ambient[] = { 0.3f, 0.3f, 0.3f, 1.0f };
    float light_diffuse[] = { 0.7f, 0.7f, 0.7f, 1.0f };
    float light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };

    glLightfv(GL_LIGHT0, GL_POSITION, light_position1);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glEnable(GL_LIGHT0);

    glLightfv(GL_LIGHT1, GL_POSITION, light_position2);
    glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT1, GL_SPECULAR, light_specular);
    glEnable(GL_LIGHT1);

    // initialize material
    float specref[] = { 1.0f, 1.0f, 1.0f, 1.0f };

    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specref);
    glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, 128);
    glEnable(GL_COLOR_MATERIAL);
}

void VisualizerWidget::resizeGL(int w, int h)
{
}

void VisualizerWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);
}

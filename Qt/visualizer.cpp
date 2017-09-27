#include "visualizer.h"

void VisualizerWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glShadeModel(GL_SMOOTH);
    glDepthRange(0.0, 1.0);

    //load Texture()
    QPixmap data_image("particle.png");

    //Make a power of two version
    size_t u2 = 1; while (u2 < data_image.width()) u2 *= 2;
    size_t v2 = 1; while (v2 < data_image.height()) v2 *= 2;

    // Make power of two version of the image.
    data_image.scaled(u2,v2);

    glGenTextures(1, &textures);
    glBindTexture(GL_TEXTURE_2D, textures);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, 4, u2, v2, 0, GL_RGBA, GL_UNSIGNED_BYTE, data_image.toImage().bits());

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

void VisualizerWidget::drawGraph(Graph *g){
    glVertexPointer(3, GL_FLOAT, 0, g->coords);
    glEnableClientState(GL_VERTEX_ARRAY);

    glEnable(GL_BLEND);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glDisable(GL_LIGHTING);
    glLineWidth(0.001f);

    glDrawElements(GL_LINES, g->edges().size() * 2, GL_UNSIGNED_INT, g->edges());
    glDepthMask(GL_TRUE);

    glBindTexture(GL_TEXTURE_2D, textures);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);


    glPointSize(4.3f);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glDrawArrays(GL_POINTS, 0, g->nodes().size());
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_TRUE);

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_POINT_SPRITE);
    glDisable(GL_BLEND);

    glDisable(GL_BLEND);

}

void VisualizerWidget::drawAlignment(Alignment *align){



}

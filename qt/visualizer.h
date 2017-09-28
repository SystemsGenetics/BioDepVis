#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include "alignment.h"
#include "database.h"
#include "graph.h"

class Visualizer : public QOpenGLWidget, protected QOpenGLFunctions
{
private:
    GLuint _textures;

    Database *_db;
    bool _animate;
    bool _showalignment;
    bool _showcluster;

public:
    Visualizer(Database *db, QWidget *parent=Q_NULLPTR);

private:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();

    void draw_graph(const Graph& g, const color_t& edge_color);
    void draw_alignment(Alignment& a, const color_t& edge_color);
};

#endif

#include "visualizer.h"

VisualizerWidget::VisualizerWidget(Database *db, QWidget *parent)
    : QOpenGLWidget(parent)
{
    this->_db = db;
    this->_animate = false;
    this->_showalignment = false;
}

void VisualizerWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glShadeModel(GL_SMOOTH);
    glDepthRange(0.0, 1.0);

    //load Texture()
    QImage image("particle.png");

    // scale each image dimension to next power of two
    int u2 = 1;
    int v2 = 1;
    while ( u2 < image.width() ) u2 *= 2;
    while ( v2 < image.height() ) v2 *= 2;

    image = image.scaled(u2, v2);

    glGenTextures(1, &_textures);
    glBindTexture(GL_TEXTURE_2D, _textures);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, 4, u2, v2, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.bits());

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
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // TODO: camera->PerspectiveDisplay

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for ( Alignment& a : _db->alignments() ) {
        this->draw_alignment(a);
    }

    for ( Graph *g : _db->graphs() ) {
        this->draw_graph(*g);
    }
}

void VisualizerWidget::draw_graph(const Graph& g)
{
    glVertexPointer(3, GL_FLOAT, 0, g.coords().data());
    glEnableClientState(GL_VERTEX_ARRAY);

    glEnable(GL_BLEND);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glDisable(GL_LIGHTING);
    glLineWidth(0.001f);

    glDrawElements(GL_LINES, g.edges().size() * sizeof(graph_edge_t), GL_UNSIGNED_INT, g.edges().data());
    glDepthMask(GL_TRUE);

    glBindTexture(GL_TEXTURE_2D, _textures);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

    glPointSize(4.3f);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glDrawArrays(GL_POINTS, 0, g.nodes().size());
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_TRUE);

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_POINT_SPRITE);
    glDisable(GL_BLEND);

    glDisable(GL_BLEND);
}

void VisualizerWidget::draw_alignment(Alignment& a)
{
    glDepthMask(GL_FALSE);

    if ( _animate ) {
        a.update();
    }

    if ( _showalignment ) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glLineStipple(1, 0xAAAA);
        // glEnable(GL_LINE_STIPPLE);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, 0, a.vertices().data());
        glColor4f(0.3f, 0.3f, 0.3f, 0.5f);
        glColor4f(0.69f, 0.19f, 0.29f, 0.005f);
        // TODO: edgeAlignmentColor

        glLineWidth(0.1);
        glDrawArrays(GL_LINES, 0, a.edges().size());
        glDisable(GL_BLEND);
        glDisable(GL_LINE_STIPPLE);
        glDisableClientState(GL_VERTEX_ARRAY);
    }

    if ( !_animate && !_showalignment ) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glColor4f(0.69f, 0.19f, 0.29f, 0.005f);

        glEnable(GL_MAP1_VERTEX_3);

        for ( int k = 0; k < a.vertices().size(); k++ ) {
            const vec3_t& v1 = a.vertices()[k].v1;
            const vec3_t& v2 = a.vertices()[k].v2;

            vec3_t C1 = {
                (v1.x + v2.x) / 2.0f,
                (v1.y + v2.y) / 2.0f,
                (v1.z + v2.z) / 10.0f - 150.0f
            };
            vec3_t C2 = {
                (v1.x + v2.x) / 2.0f,
                (v1.y + v2.y) / 2.0f,
                (v1.z + v2.z) * 9 / 10.0f - 150.0f
            };
            vec3_t ctrl_points[4] = { v1, C1, C2, v2 };

            glMap1f(GL_MAP1_VERTEX_3, 0.0f, 1.0f, 3, 4, (float *)ctrl_points);

            glLineWidth(0.1);

            glBegin(GL_LINE_STRIP);
            for ( int i = 0; i <= 20; i++ ) {
                glEvalCoord1f(i / 20.0f);
            }
            glEnd();
        }

        glDisable(GL_BLEND);
        glDisable(GL_MAP1_VERTEX_3);
    }
    glDepthMask(GL_TRUE);
}

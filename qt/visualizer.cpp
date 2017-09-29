#include "visualizer.h"

#define NORMALIZE(v) ((float)(v) / 255)

const color_t GRAPH_EDGE_COLORS[] = {
    { NORMALIZE(166), NORMALIZE(206), NORMALIZE(227), 0.10f },
    { NORMALIZE(126), NORMALIZE(127), NORMALIZE(200), 0.10f },
    { NORMALIZE(106), NORMALIZE(61),  NORMALIZE(154), 0.10f },
    { NORMALIZE(31),  NORMALIZE(120), NORMALIZE(180), 0.10f },
    { NORMALIZE(178), NORMALIZE(223), NORMALIZE(138), 0.10f },
    { NORMALIZE(255), NORMALIZE(127), NORMALIZE(0),   0.10f }
};

const color_t ALIGN_EDGE_COLORS[] = {
    { NORMALIZE(189), NORMALIZE(63),  NORMALIZE(243), 0.05f },
    { NORMALIZE(226), NORMALIZE(127), NORMALIZE(202), 0.05f },
    { NORMALIZE(226), NORMALIZE(127), NORMALIZE(202), 0.05f }
};

Visualizer::Visualizer(Database *db, QWidget *parent)
    : QOpenGLWidget(parent)
{
    this->_camera = Camera(
        { 0, 10, 400 },
        { 0, 0, 0 },
        { 0, 1, 0 }
    );
    this->_db = db;
    this->_animate = false;
    this->_showalignment = false;
    this->_showcluster = false;
}

void Visualizer::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glShadeModel(GL_SMOOTH);
    glDepthRange(0.0, 1.0);

    // load texture image
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
    float light_position1[] = { 2.0f, -200.0f, 0.0f, 0.0f };
    float light_position2[] = { 0.0f, 10.0f, -5.0f, 0.0f };
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

void Visualizer::resizeGL(int w, int h)
{
}

void Visualizer::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    this->_camera.display(this->width(), this->height());

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for ( int i = 0; i < _db->graphs().size(); i++ ) {
        Graph *g = _db->graphs().values()[i];

        this->draw_graph(*g, GRAPH_EDGE_COLORS[i]);
    }

    for ( int i = 0; i < _db->alignments().size(); i++ ) {
        Alignment *a = _db->alignments()[i];

        this->draw_alignment(*a, ALIGN_EDGE_COLORS[i]);
    }
}

void Visualizer::draw_graph(const Graph& g, const color_t& edge_color)
{
    glVertexPointer(3, GL_FLOAT, 0, g.coords().data());
    glEnableClientState(GL_VERTEX_ARRAY);

    glEnable(GL_BLEND);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glDisable(GL_LIGHTING);
    glLineWidth(0.001f);
    glColor4f(edge_color.r, edge_color.g, edge_color.b, edge_color.a);
    glDrawElements(GL_LINES, g.edges().size() * sizeof(graph_edge_t), GL_UNSIGNED_INT, g.edges().data());
    glDepthMask(GL_TRUE);

    glColor4f(0.0f, 0.0f, 0.0f, 0.0f);

    if ( _showcluster ) {
        glColorPointer(4, GL_FLOAT, 0, g.colors().data());
        glEnableClientState(GL_COLOR_ARRAY);
    }

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

void Visualizer::draw_alignment(Alignment& a, const color_t& edge_color)
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
        glColor4f(edge_color.r, edge_color.g, edge_color.b, edge_color.a);

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

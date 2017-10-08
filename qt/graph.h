#ifndef GRAPH_H
#define GRAPH_H

#include <QString>
#include <QVector>
#include "matrix.h"
#include "vector.h"

typedef struct {
    float r;
    float g;
    float b;
    float a;
} color_t;

typedef struct {
    QString name;
    int module_id;
    QStringList go_terms;
} graph_node_t;

typedef struct {
    int node1;
    int node2;
} graph_edge_t;

class Graph
{
private:
    int _id;
    QString _name;
    vec3_t _center;
    float _width;
    float _height;

    QVector<graph_node_t> _nodes;
    QVector<graph_edge_t> _edges;
    QVector<color_t> _colors;

    QVector<vec3_t> _coords;
    QVector<vec3_t> _coords_d;
    Matrix _edge_matrix;

    vec3_t *_coords_gpu;
    vec3_t *_coords_d_gpu;
    int *_edge_matrix_gpu;

public:
    Graph(
        int id, const QString& name,
        const QString& nodefile,
        const QString& edgefile,
        const QString& ontfile,
        float x, float y, float z, float w, float h
    );
    Graph() {};
    ~Graph();

    int id() const { return this->_id; }
    const QString& name() const { return this->_name; }
    const QVector<graph_node_t>& nodes() const { return this->_nodes; }
    const QVector<graph_edge_t>& edges() const { return this->_edges; }
    const QVector<color_t>& colors() const { return this->_colors; }

    QVector<vec3_t>& coords() { return this->_coords; }
    QVector<vec3_t>& coords_d() { return this->_coords_d; }
    const Matrix& edge_matrix() const { return this->_edge_matrix; }

    vec3_t * coords_gpu() { return this->_coords_gpu; }
    vec3_t * coords_d_gpu() { return this->_coords_d_gpu; }
    int * edge_matrix_gpu() const { return this->_edge_matrix_gpu; }

    int find_node(const QString& name);

    void load_nodes(const QString& filename);
    void load_edges(const QString& filename);
    void load_ontology(const QString& filename);

    void print() const;
};

#endif // GRAPH

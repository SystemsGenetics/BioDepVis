#ifndef GRAPH_H
#define GRAPH_H

#include <QHash>
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
} node_t;

typedef struct {
    int node1;
    int node2;
} edge_idx_t;

class Graph
{
private:
    int _id;
    QString _name;
    vec3_t _center;
    float _width;
    float _height;

    QVector<node_t> _nodes;
    QHash<QString, int> _node_map;
    QVector<edge_idx_t> _edges;
    QVector<color_t> _colors;

    QVector<vec3_t> _positions;
    QVector<vec3_t> _positions_d;
    Matrix _edge_matrix;

    vec3_t *_positions_gpu;
    vec3_t *_positions_d_gpu;
    bool *_edge_matrix_gpu;

public:
    Graph(
        int id, const QString& name,
        const QString& nodefile,
        const QString& edgefile,
        const QString& ontfile,
        float x, float y, float z, float w, float h
    );
    Graph();
    ~Graph();

    int id() const { return this->_id; }
    const QString& name() const { return this->_name; }
    QVector<node_t>& nodes() { return this->_nodes; }
    QVector<edge_idx_t>& edges() { return this->_edges; }
    QVector<color_t>& colors() { return this->_colors; }

    QVector<vec3_t>& positions() { return this->_positions; }
    QVector<vec3_t>& positions_d() { return this->_positions_d; }
    Matrix& edge_matrix() { return this->_edge_matrix; }

    vec3_t * positions_gpu() { return this->_positions_gpu; }
    vec3_t * positions_d_gpu() { return this->_positions_d_gpu; }
    bool * edge_matrix_gpu() const { return this->_edge_matrix_gpu; }

    void read_gpu();
    void write_gpu();

    int find_node(const QString& name);

    void init_node_map();
    void load_nodes(const QString& filename);
    void load_edges(const QString& filename);
    void load_ontology(const QString& filename);

    void save_nodes(const QString& filename);
    void save_edges(const QString& filename);

    void print() const;
};

#endif // GRAPH

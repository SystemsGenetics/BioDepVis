#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include "graph.h"

typedef struct {
    vec3_t v1;
    vec3_t v2;
} align_edge_t;

class Alignment
{
private:
    Graph *_graph1;
    Graph *_graph2;
    QVector<graph_edge_t> _edges;
    QVector<align_edge_t> _vertices;
    Matrix _edge_matrix;

public:
    Alignment(const QString& filename, Graph *graph1, Graph *graph2);
    Alignment();
    ~Alignment() {};

    const QVector<graph_edge_t>& edges() const { return this->_edges; }
    const QVector<align_edge_t>& vertices() const { return this->_vertices; }

    void load_edges(const QString& filename);
    void update();

    void print() const;
};

#endif // ALIGNMENT_H

#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include <QPair>
#include "graph.h"

typedef struct {
    vec3_t v1;
    vec3_t v2;
} edge_t;

class Alignment
{
private:
    Graph *_graph1;
    Graph *_graph2;
    QVector<QPair<int, int>> _edges;

    int _rows;
    int _cols;
    float *_edge_matrix;
    edge_t *_vertices;

public:
    Alignment(const QString& filename, Graph *graph1, Graph *graph2);
    Alignment();
    ~Alignment();

    bool load_edges(const QString& filename);
    void update();

    void print() const;
};

#endif // ALIGNMENT_H

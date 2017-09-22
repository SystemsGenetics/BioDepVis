#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include <QPair>
#include "graph.h"

class Alignment
{
private:
    Graph *_graph1;
    Graph *_graph2;
    QVector<QPair<int, int>> _edges;

    int _rows;
    int _cols;
    float *_edge_matrix;

public:
    Alignment(const QString& filename, Graph *graph1, Graph *graph2);
    Alignment();
    ~Alignment();

    bool load_edges(const QString& filename);

    void print() const;
};

#endif // ALIGNMENT_H

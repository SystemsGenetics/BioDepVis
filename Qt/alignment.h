#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include "graph.h"

class Alignment
{
private:
    Graph *_graph1;
    Graph *_graph2;
    //std::vector<int> graph1_vertices;
    //std::vector<int> graph2_vertices;

public:
    Alignment(const QString& filename, Graph *graph1, Graph *graph2);
    Alignment() {};

    bool load_alignment(const QString& filename);

    void print() const;
};

#endif // ALIGNMENT_H

#ifndef ALIGNMENT_H
#define ALIGNMENT_H
#include "graph.h"

class Alignment
{
public:
    graph *g1;
    graph *g2;
    void run();
    void update();
	void cleanup();
    float *edgeAlignMatrix;
    float *edgeAlignMatrix_d;
    int rows;
    int cols;
    int edges;
    float *vertices;
    std::vector <int> g1_vertices;
    std::vector <int>  g2_vertices;
    void readFile(char *);
    Alignment(char *,graph*,graph*);

};

#endif // ALIGNMENT_H

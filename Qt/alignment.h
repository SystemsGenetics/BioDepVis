#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include "graph.h"

class Alignment
{
private:
    int _id1;
    int _id2;

public:
    Alignment(const QString& filename, int id1, int id2);

};

#endif // ALIGNMENT_H

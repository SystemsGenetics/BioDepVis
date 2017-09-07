#ifndef GRAPH_H
#define GRAPH_H

#include <QString>

class Graph
{
private:
    int _id;
    QString _name;

public:
    Graph(
        int id, const QString& name,
        const QString& datafile,
        const QString& clusterfile,
        const QString& ontfile,
        int x, int y, int z, int w, int h
    );

    int id() const { return this->_id; }
    const QString& name() const { return this->_name; }
};

#endif // GRAPH

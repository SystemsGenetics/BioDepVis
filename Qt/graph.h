#ifndef GRAPH_H
#define GRAPH_H

#include <QString>
#include <QVector>

typedef struct {
    float dx;
    float dy;
    float dz;
    float radius;
    int clusterId;
} coord_t;

typedef struct {
    QString name;
    QStringList go_terms;
} node_t;

class Graph
{
private:
    QString _name;
    int _id;
    int nodes;
    int edges;
    float *edgeMatrix;
    QVector<node_t> nodeList;
    QVector<coord_t> coinfo;

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
    const QVector<node_t>& node_list() const { return this->nodeList; }

    bool load_datafile(const QString& filename);
    bool load_clusterfile(const QString& filename);
    bool load_ontologyfile(const QString& filename);

    void print() const;
};

#endif // GRAPH

#ifndef GRAPH_H
#define GRAPH_H

#include <QString>
#include <QVector>

typedef struct {
    float dx;
    float dy;
    float dz;
    float radius;
} coinfo_t;

typedef struct {
    QString name;
    int cluster_id;
    QStringList go_terms;
    coinfo_t coinfo;
} node_t;

class Graph
{
private:
    int _id;
    QString _name;
    QVector<node_t> _nodes;
    float *_edge_matrix;
    int _num_edges;

public:
    Graph(
        int id, const QString& name,
        const QString& datafile,
        const QString& clusterfile,
        const QString& ontfile,
        int x, int y, int z, int w, int h
    );
    Graph() {};

    int id() const { return this->_id; }
    const QString& name() const { return this->_name; }
    const QVector<node_t>& nodes() const { return this->_nodes; }

    bool load_datafile(const QString& filename);
    bool load_clusterfile(const QString& filename);
    bool load_ontologyfile(const QString& filename);

    void print() const;
};

#endif // GRAPH

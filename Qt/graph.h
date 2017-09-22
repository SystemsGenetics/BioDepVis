#ifndef GRAPH_H
#define GRAPH_H

#include <QPair>
#include <QString>
#include <QVector>

typedef struct {
    float x;
    float y;
    float z;
} vec3_t;

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
    vec3_t coord;
    coinfo_t coinfo;
} node_t;

int find_node(const QVector<node_t>& nodes, const QString& name);

class Graph
{
private:
    int _id;
    QString _name;
    QVector<node_t> _nodes;
    QVector<QPair<int, int>> _edges;

    float *_edge_matrix;

    vec3_t _center;
    int _width;
    int _height;

public:
    Graph(
        int id, const QString& name,
        const QString& datafile,
        const QString& clusterfile,
        const QString& ontfile,
        float x, float y, float z, int w, int h
    );
    Graph();
    ~Graph();

    int id() const { return this->_id; }
    const QString& name() const { return this->_name; }
    const QVector<node_t>& nodes() const { return this->_nodes; }

    bool load_clusterfile(const QString& filename);
    bool load_edgefile(const QString& filename);
    bool load_ontologyfile(const QString& filename);

    void print() const;
};

#endif // GRAPH

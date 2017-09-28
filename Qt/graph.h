#ifndef GRAPH_H
#define GRAPH_H

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
} graph_node_t;

typedef struct {
    int node1;
    int node2;
} graph_edge_t;

int find_node(const QVector<graph_node_t>& nodes, const QString& name);

class Graph
{
private:
    int _id;
    QString _name;
    vec3_t _center;
    int _width;
    int _height;

    QVector<graph_node_t> _nodes;
    QVector<vec3_t> _coords;
    QVector<coinfo_t> _coinfo;

    QVector<graph_edge_t> _edges;
    float *_edge_matrix;

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
    const QVector<graph_node_t>& nodes() const { return this->_nodes; }
    const QVector<graph_edge_t>& edges() const { return this->_edges; }

    const QVector<vec3_t>& coords() const { return this->_coords; }
    const QVector<coinfo_t>& coinfo() const { return this->_coinfo; }

    void load_clusterfile(const QString& filename);
    void load_edgefile(const QString& filename);
    void load_ontologyfile(const QString& filename);

    void print() const;
};

#endif // GRAPH

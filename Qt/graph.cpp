#include "graph.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>

/**
 * Find a node by name.
 *
 * @param nodes
 * @param name
 */
int find_node(const QVector<node_t>& nodes, const QString& name)
{
    for ( int i = 0; i < nodes.size(); i++ ) {
        if ( nodes[i].name == name ) {
            return i;
        }
    }

    return -1;
}

Graph::Graph(
    int id, const QString& name,
    const QString& datafile,
    const QString& clusterfile,
    const QString& ontfile,
    int x, int y, int z, int w, int h)
{
    this->_id = id;
    this->_name = name;

    load_datafile(datafile);
    load_clusterfile(clusterfile);
    load_ontologyfile(ontfile);

    // initialize node coordinates
    for ( node_t& node : this->_nodes ) {
        node.coinfo = { 0, 0, 0, 1 };
    }
}

Graph::Graph()
{
    this->_edge_matrix = nullptr;
}

Graph::~Graph()
{
    delete[] this->_edge_matrix;
}

bool Graph::load_datafile(const QString& filename)
{
    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    // read node pairs from file
    QTextStream in(&file);
    QVector<QPair<QString, QString>> node_pairs;

    while ( !in.atEnd() ) {
        QStringList list = in.readLine().split("\t");
        QString node1 = list[0];
        QString node2 = list[1];

        node_pairs.push_back({ node1, node2 });
    }

    // construct node list from pairs
    for ( auto& p : node_pairs ) {
        QString node1 = p.first;
        QString node2 = p.second;

        if ( find_node(this->_nodes, node1) == -1 ) {
            node_t n;
            n.name = node1;

            this->_nodes.push_back(n);
        }

        if ( find_node(this->_nodes, node2) == -1 ) {
            node_t n;
            n.name = node2;

            this->_nodes.push_back(n);
        }
    }

    // construct edge matrix
    int n = this->_nodes.size();

    this->_edge_matrix = new float[n * n];
    this->_num_edges = 0;

    for ( auto& p : node_pairs ) {
        QString node1 = p.first;
        QString node2 = p.second;
        int i = find_node(this->_nodes, node1);
        int j = find_node(this->_nodes, node2);

        this->_edge_matrix[i * n + j] = 1;
        this->_edge_matrix[j * n + i] = 1;

        this->_num_edges++;
    }

    return true;
}

bool Graph::load_clusterfile(const QString& filename)
{
    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    QTextStream in(&file);

    while ( !in.atEnd() ) {
        QStringList list = in.readLine().split("\t");
        QString node = list[0];
        int cluster_id = list[1].toInt();

        int nodeIndex = find_node(this->_nodes, node);
        this->_nodes[nodeIndex].cluster_id = cluster_id;
    }

    return true;
}

bool Graph::load_ontologyfile(const QString& filename)
{
    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    QTextStream in(&file);

    while ( !in.atEnd() ) {
        QStringList fields = in.readLine().split("\t");
        QString name = fields[1];
        QStringList go_terms = fields[9].split(",");

        int nodeIndex = find_node(this->_nodes, name);

        if ( nodeIndex != -1 ) {
            this->_nodes[nodeIndex].go_terms = go_terms;
        }
    }

    return true;
}

void Graph::print() const
{
    qDebug() << this->_id << this->_name;

    for ( int i = 0; i < this->_nodes.size(); i++ ) {
        qDebug()
            << this->_nodes[i].name
            << this->_nodes[i].cluster_id
            << this->_nodes[i].go_terms.join(' ');
    }
}

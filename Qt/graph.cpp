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
    float x, float y, float z, int w, int h)
{
    this->_id = id;
    this->_name = name;
    this->_center = { x, y, z };
    this->_width = w;
    this->_height = h;

    load_clusterfile(clusterfile);
    load_edgefile(datafile);
    load_ontologyfile(ontfile);

    // initialize node coords, coinfo
    for ( node_t& node : this->_nodes ) {
        node.coord = {
            x - w/2 + qrand() % w,
            y - w/2 + qrand() % h,
            z
        };

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

bool Graph::load_clusterfile(const QString& filename)
{
    qDebug() << "- loading cluster file...";

    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    QTextStream in(&file);

    while ( !in.atEnd() ) {
        QStringList list = in.readLine().split("\t");
        QString name = list[0];
        int cluster_id = list[1].toInt();

        node_t node;
        node.name = name;
        node.cluster_id = cluster_id;

        this->_nodes.push_back(node);
    }

    return true;
}

bool Graph::load_edgefile(const QString& filename)
{
    qDebug() << "- loading edge list...";

    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    // load edges from file
    QTextStream in(&file);

    while ( !in.atEnd() ) {
        QStringList list = in.readLine().split("\t");
        QString node1 = list[0];
        QString node2 = list[1];

        int i = find_node(this->_nodes, node1);
        int j = find_node(this->_nodes, node2);

        if ( i != -1 && j != -1 ) {
            this->_edges.push_back({ i, j });
        }
        else {
            qDebug() << "warning: could not find nodes " << node1 << node2;
        }
    }

    // initialize edge matrix
    int rows = this->_nodes.size();

    this->_edge_matrix = new float[rows * rows];

    memset(this->_edge_matrix, 0, rows * rows * sizeof(float));

    for ( auto& edge : this->_edges ) {
        int i = edge.first;
        int j = edge.second;

        this->_edge_matrix[i * rows + j] = 1;
        this->_edge_matrix[j * rows + i] = 1;
    }

    return true;
}

bool Graph::load_ontologyfile(const QString& filename)
{
    qDebug() << "- loading ontology file...";

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

    // for ( int i = 0; i < this->_nodes.size(); i++ ) {
    //     qDebug()
    //         << this->_nodes[i].name
    //         << this->_nodes[i].cluster_id
    //         << this->_nodes[i].go_terms.join(' ');
    // }
}
